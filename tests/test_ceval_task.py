import pytest

@pytest.mark.usefixtures('task')
@pytest.mark.parametrize("task", ("ceval",), indirect=True)
class TestCevalTask:
    completations = [
        '根据假说一演绎法的基本思路，孟德尔在发现基因分离定律时的“演绎推理”内容应该是B. 由F2出现了“3：1”的表现型比，推测生物体产生配子时，成对遗传因子彼此分离。这个推理是基于孟德尔对豌豆杂交实验结果的观察和统计分析，得出了基因分离定律。',
    ]
    acompletations = [
        '计算机网络的资源主要是指可以被共享和利用的各种设备、软件和数据等。因此，选项A中的服务器、路由器、通信线路与用户计算机是计算机网络中的资源，选项B中的计算机操作系统、数据库与应用软件是计算机系统中的资源，选项C中的计算机硬件、软件与数据是计算机系统中的资源，选项D中的Web服务器、数据库服务器与文件服务器是网络中的服务器资源。综上所述，选项A是正确答案',
    ]

    def test_extract(self, task):
        labels = ['A']
        ans = []
        for completation in self.completations:
            a = task.extract(completation)
            ans.append(task.extract(completation))
        assert ans == labels
