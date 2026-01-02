"""
语言模型适配器模块
"""

import logging
import time
from typing import Optional
import dspy
import ollama
import requests

from src.config import config
from src.utils import retry_on_failure

logger = logging.getLogger(__name__)


class OllamaLM(dspy.LM):
    """Ollama语言模型适配器"""

    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None):
        selected_model = model or config.ollama_model
        super().__init__(selected_model)
        self.model = selected_model
        self.base_url = base_url or config.ollama_base_url

        # 测试连接
        try:
            response = ollama.list()
            print(
                f"✅ 连接到 Ollama，可用模型: {[m['name'] for m in response['models']]}"
            )
        except Exception as e:
            raise ConnectionError(f"无法连接到 Ollama ({self.base_url}): {e}")

    def basic_request(self, prompt: str, **kwargs):
        """基础请求方法"""
        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            options={
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 1000),
                "top_p": kwargs.get("top_p", 0.9),
            },
        )
        return response

    def __call__(
        self,
        prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ):
        response = self.basic_request(
            prompt or "",
            max_tokens=max_tokens,
            **kwargs,
        )
        return [response["response"]]


class RemoteOllamaLM(dspy.LM):
    """远程Ollama语言模型适配器"""

    def __init__(
        self,
        model: str = "",
        base_url: str = "",
        timeout: int = 30,
    ):
        super().__init__(model)
        self.model = model or config.ollama_model
        self.base_url = base_url or config.ollama_base_url
        self.timeout = timeout or config.ollama_timeout

        # 配置ollama客户端
        self._configure_ollama_client()

        # 测试连接
        self._test_connection()

    def _configure_ollama_client(self):
        """配置Ollama客户端"""
        # 创建ollama客户端实例
        self.client = ollama.Client(host=self.base_url, timeout=self.timeout)

        logger.info(f"配置Ollama客户端: {self.base_url}, 模型: {self.model}")

    @retry_on_failure(max_retries=config.ollama_max_retries)
    def _test_connection(self):
        """测试连接和模型可用性"""
        try:
            logger.info("测试Ollama连接...")
            logger.info(f"API URL: {self.base_url}")

            # 测试API连接
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            logger.info(f"响应状态码: {response.status_code}")
            logger.info(f"响应内容: {response.text[:100]}...")
            response.raise_for_status()

            # 检查模型
            models = self.client.list()
            logger.debug(f"models response type: {type(models)}")
            logger.debug(f"models response: {models}")

            # Handle different response structures
            available_models = []
            if hasattr(models, "models"):
                # Response object with models attribute
                available_models = [
                    m.get("name", m.get("model", "")) for m in models.models
                ]
            elif isinstance(models, dict) and "models" in models:
                # Dict response
                available_models = [
                    m.get("name", m.get("model", "")) for m in models["models"]
                ]

            # Filter out empty strings
            available_models = [m for m in available_models if m]

            logger.info(f"✅ 连接成功! 可用模型: {available_models}")

            # 检查所需模型是否可用
            if self.model not in available_models:
                logger.warning(
                    f"⚠️  模型 {self.model} 未找到，可用模型: {available_models}"
                )
                logger.warning(f"建议运行: ollama pull {self.model}")

            # 显示模型详情
            try:
                model_info = self.client.show(self.model)
                logger.info(
                    f"📋 模型详情: {model_info.get('modelfile', 'N/A')[:100]}..."
                )
            except:
                logger.warning(f"无法获取模型 {self.model} 的详细信息")

        except requests.exceptions.ConnectionError as e:
            error_msg = (
                f"❌ 无法连接到Ollama服务器 {self.base_url}\n"
                f"   请检查:\n"
                f"   1. 服务器地址是否正确\n"
                f"   2. 服务器是否正在运行\n"
                f"   3. 防火墙/网络设置\n"
                f"   4. 如果需要认证，请设置 OLLAMA_USERNAME 和 OLLAMA_PASSWORD"
            )
            logger.error(error_msg)
            raise ConnectionError(error_msg) from e
        except Exception as e:
            logger.error(f"连接测试失败: {e}")
            raise

    @retry_on_failure(max_retries=config.ollama_max_retries)
    def basic_request(self, prompt: str | None = None, **kwargs):
        """基础请求方法 - 支持远程调用"""
        if prompt is None:
            prompt = ""
        try:
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "temperature": kwargs.get("temperature", 0.7),
                    "max_tokens": kwargs.get("max_tokens", 1000),
                    "top_p": kwargs.get("top_p", 0.9),
                    "top_k": kwargs.get("top_k", 40),
                },
                stream=False,
            )
            return response
        except ollama.ResponseError as e:
            logger.error(f"Ollama API错误: {e.error}")
            if "model not found" in str(e.error).lower():
                raise ValueError(f"模型 {self.model} 未找到，请先在服务器上安装")
            raise
        except Exception as e:
            logger.error(f"请求失败: {e}")
            raise

    def __call__(
        self,
        prompt: Optional[str] = None,
        messages: Optional[list] = None,
        **kwargs,
    ):
        """调用语言模型"""
        start_time = time.time()

        # Log all parameters received
        logger.info(f"LM __call__ 被调用")
        logger.info(
            f"  - prompt类型: {type(prompt)}, 长度: {len(prompt) if prompt else 0}"
        )
        logger.info(f"  - messages: {messages}")
        logger.info(f"  - kwargs: {list(kwargs.keys())}")

        # DSPy may pass the prompt in different ways
        actual_prompt = prompt

        # Check if prompt is in kwargs
        if not actual_prompt and "prompt" in kwargs:
            actual_prompt = kwargs.pop("prompt")
            logger.info(
                f"从kwargs中获取prompt, 长度: {len(actual_prompt) if actual_prompt else 0}"
            )

        # Check if messages format is used
        if not actual_prompt and messages:
            # Convert messages to prompt
            actual_prompt = "\n".join(
                [m.get("content", "") for m in messages if isinstance(m, dict)]
            )
            logger.info(f"从messages转换prompt, 长度: {len(actual_prompt)}")

        if not actual_prompt:
            logger.error("⚠️ 没有收到有效的prompt!")
            logger.error(f"完整kwargs: {kwargs}")
            # Return empty to avoid crash
            return [""]

        logger.info(f"最终prompt长度: {len(actual_prompt)}")
        logger.debug(f"Prompt内容: {actual_prompt[:200]}...")

        try:
            response = self.basic_request(actual_prompt, **kwargs)
            elapsed = time.time() - start_time

            # Extract response text
            response_text = response.get("response", "")
            logger.info(f"模型响应时间: {elapsed:.2f}s, 响应长度: {len(response_text)}")
            logger.debug(f"响应内容: {response_text[:200]}...")

            if not response_text:
                logger.warning("⚠️ 模型返回空响应!")

            return [response_text]
        except Exception as e:
            logger.error(f"模型调用失败: {e}", exc_info=True)
            raise

    def stream_generate(self, prompt: str, **kwargs):
        """流式生成（可选功能）"""
        try:
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                stream=True,
                options={
                    "temperature": kwargs.get("temperature", 0.7),
                    "max_tokens": kwargs.get("max_tokens", 1000),
                },
            )

            for chunk in response:
                if chunk.get("response"):
                    yield chunk["response"]
        except Exception as e:
            logger.error(f"流式生成失败: {e}")
            raise
