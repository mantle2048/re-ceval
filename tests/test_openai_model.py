import pytest
import asyncio


@pytest.mark.usefixtures("model", 'task')
@pytest.mark.parametrize(
    "model",
    [
        ("openai", "gpt-3.5-turbo"),
        # ("openai", "gpt-4"),
        # ("openai", "text-davinci-003")
    ],
    indirect=True
)
@pytest.mark.parametrize("task", ("ceval",), indirect=True)
class TestOpenAIModel:
    datum = {
        'id': 0,
        'question': '下列设备属于资源子网的是____。',
        'A': '计算机软件',
        'B': '网桥',
        'C': '交换机',
        'D': '路由器',
        'answer': 'A',
        'explanation': '1. 首先，资源子网是指提供共享资源的网络，如打印机、文件服务器等。\r\n2. 其次，我们需要了解选项中设备的功能 。网桥、交换机和路由器的主要功能是实现不同网络之间的通信和数据传输，是通信子网设备。而计算机软件可以提供共享资源的功能。'
    }

    def test_compute_cost(self, model):
        prompt_tokens = 2000
        completion_tokens = 4000
        cost = model.compute_cost(prompt_tokens, completion_tokens)
        if model.name == 'gpt-3.5-turbo':
            assert 0.012 == pytest.approx(cost, 1e-6)
        elif model.name == 'gpt-4':
            assert 0.3 == pytest.approx(cost, 1e-6)

    def test_count_tokens(self, model):
        content = "Follow the given examples and answer the question."
        count = model.count_tokens(content)
        assert count == 9

    def test_format(self, model, task):
        prompt = task.format(self.datum)
        if task.prompt_type == 'vanilla':
            assert prompt == '下列设备属于资源子网的是____。\nA. 计算机软件\nB. 网桥\nC. 交换机\nD. 路由器\n答案：'
        elif task.prompt_type == 'cot':
            assert prompt == '下列设备属于资源子网的是____。\nA. 计算机软件\nB. 网桥\nC. 交换机\nD. 路由器\n答案：让我们一步一步思考，\n'

    def test_complete(self, model, task):
        prompt = task.prompt + task.format(self.datum)
        result = model.complete(
            prompt,
            system_message=task.system_message
        )
        answer = task.extract(result.text)
        print(result)
        print(answer)
        assert answer in ('A', 'B', 'C', 'D', None)

    def test_acomplete(self, model, task):
        prompt = task.prompt + task.format(self.datum)
        coroutine = model.acomplete(prompt, system_message=task.system_message)
        result = asyncio.run(coroutine)
        answer = task.extract(result.text)
        print(result)
        print(answer)
        assert answer in ('A', 'B', 'C', 'D', None)
