import pytest

@pytest.mark.usefixtures("model", 'task')
@pytest.mark.parametrize(
    "model",
    [
        ("llama", "llama-7b-hf"),
    ],
    indirect=True
)
@pytest.mark.parametrize("task", ("ceval",), indirect=True)
class TestLLaMAModel:

    prompt = "Q: Which sentence has the correct adjective order:\nOptions:\n(A) blue gold wonderful square shoe\n(B) wonderful square blue gold shoe\nA: Let's think step by step."
    content = '根据假说一演绎法的基本思路，孟德尔在发现基因分离定律时的“演绎推理”内容应该是B. 由F2出现了“3：1”的表现型比，推测生物体产生配子时，成对遗传因子彼此分离。这个推理是基于孟德尔对豌豆杂交实验结果的观察和统计分析，得出了基因分离定律。',

    def est_count_tokens(self, model):
        length = model.count_tokens(self.content)
        assert length == 171

    def test_complete(self, model):
        import time
        start = time.time()
        result = model.complete(self.prompt)
        print(time.time() - start)
        print(result.text)
