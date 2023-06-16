import pytest

@pytest.mark.usefixtures("model", 'task')
@pytest.mark.parametrize(
    "model",
    [
        ("llama", "llama-7b"),
    ],
    indirect=True
)
@pytest.mark.parametrize("task", ("ceval",), indirect=True)
class TestLLaMAModel:

    # prompt = '假说一演绎法是现代科学研究中常用的一种科学方法，其基本思路是发现问题→提出假说→演绎推理→实验检验→得出结论。下列属于孟德尔在发现基因分离定律时的“演绎推理”内容的是____ A. 若遗传因子位于染色体上,则遗传因子在体细胞中成对存在 B. 由F2出现了“3：1”的表现型比，推测生物体产生配子时，成对遗传因子彼此分离 C. 若F1产生配子时成对遗传因子分离，则F2中三种基因型个体比接近1:2：1 D. 若F1产生配子时成对遗传因子分离,则测交后代会出现接近1：1的两种性状比 答案：让我们一步一步思考，'
    prompt = "In the following sentences, explain the antecedent of the pronoun (which thing the pronoun refers to), or state that it is ambiguous.\nSentence: The patient was referred to the specialist because he had a rare skin condition.\nOptions:\n(A) The patient had a skin condition\n(B) The specialist had a skin condition\n(C) Ambiguous"
    content = '根据假说一演绎法的基本思路，孟德尔在发现基因分离定律时的“演绎推理”内容应该是B. 由F2出现了“3：1”的表现型比，推测生物体产生配子时，成对遗传因子彼此分离。这个推理是基于孟德尔对豌豆杂交实验结果的观察和统计分析，得出了基因分离定律。',

    def est_count_tokens(self, model):
        length = model.count_tokens(self.content)
        assert length == 171

    def test_complete(self, model):
        model.complete(self.prompt)
