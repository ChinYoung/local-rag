"""
RAG优化器模块
"""

from typing import List, cast
import dspy


class RAGOptimizer:
    """RAG优化器"""

    @staticmethod
    def optimize_with_bootstrap(rag_pipeline, train_examples):
        """使用BootstrapFewShot优化"""

        class RAG(dspy.Module):
            def __init__(self):
                super().__init__()
                self.retrieve = dspy.Retrieve(k=3)
                self.generate_answer = dspy.ChainOfThought(rag_pipeline.GenerateAnswer)

            def forward(self, question):
                passages = cast(List[str], self.retrieve(question))
                context = "\n\n".join(passages) if passages else ""
                return self.generate_answer(context=context, question=question)

        # 定义评估指标
        def validate_answer(example, pred, trace=None):
            # 简单评估：检查答案是否包含关键词
            gold_answer = example.answer.lower()
            pred_answer = pred.answer.lower()

            # 计算重叠词的比例
            gold_words = set(gold_answer.split())
            pred_words = set(pred_answer.split())

            if not gold_words:
                return 0

            overlap = len(gold_words.intersection(pred_words)) / len(gold_words)
            return overlap > 0.5  # 至少50%重叠

        # 优化
        teleprompter = dspy.BootstrapFewShot(metric=validate_answer)
        optimized_rag = teleprompter.compile(RAG(), trainset=train_examples)

        return optimized_rag
