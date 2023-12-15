import torch
import torch.nn as nn 
from torchvision.models import resnet50

'''
构建端到端算法 
1.特征提取
2.注意力模块
3.尝试Transformer-based biomarker prediction from
colorectal cancer histology: A large-scale
multicentric study的算法
4.构建损失函数
5.将病人的信息全部划分
'''
'''
模型构造:
提取特征 将输入tensor view到合适维度
第二步自适应池化
第三步fc
第四步注意力
'''

#构建注意力模块
class AttentionModule(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(AttentionModule, self).__init__()
        self.attention_a = [nn.Linear(L, D), nn.Tanh()]
        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(p = 0.5))
            self.attention_b.append(nn.Dropout(p = 0.5))
        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)
    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)
        return A, x 
    
def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()
        
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)     

class E2EPrognosisNet(nn.Module):
    def __init__(self, pretrained=True, n_classes=4):
        super(E2EPrognosisNet, self).__init__()
        if pretrained:
            self.feature_extractor = resnet50(pretrained=True)
            self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-2])
        else:
            self.feature_extractor = resnet50(pretrained=False)
        self.pool = nn.AdaptiveAvgPool2d(1)
        fc = [nn.Linear(2048, 1024), nn.ReLU(), nn.Dropout(0.25),nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.25)]
        attention_net = AttentionModule(512, 256,dropout=0.25, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifier = nn.Linear(3072, n_classes)


        # self.fc = nn.Linear(2048, 1)
    
    #通过注意力函数我们可以看到对300图使用注意力最后得出的就是
    #logits logits代表未被激活的输出，也就是常说的得分
    #Y_hat得到的是logits每一行中最大的值得索引，最可能的类别
    #harards表示生存分析中的风险或事件概率
    #S计算了累计生存函数，即每个时间点上的生存概率
    def forward(self, x):
        import ipdb;ipdb.set_trace()
        batch_size = x.size(0)
        x = x.view(-1, 3, 224, 224)
        x = self.feature_extractor(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        A, x = self.attention_net(x)
        A_raw = A
        A = A.view(batch_size,-1)
        x = x.view(-1, batch_size*512)
        M = torch.mm(A, x)
        # M = torch.mm(A, x)
        logits = self.classifier(M)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        return hazards, S, Y_hat
    


if __name__ == '__main__':
    model = E2EPrognosisNet()
    x = torch.randn(6,50,3,224,224)
    out = model(x)
    import ipdb;ipdb.set_trace()