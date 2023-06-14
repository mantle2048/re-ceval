import pytest

@pytest.mark.usefixtures('task')
@pytest.mark.parametrize("task", ("ceval",), indirect=True)
class TestCevalTask:
    completations = [
        'D. 生物膜上的蛋白质具有催化、运输和识别等功能',
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
