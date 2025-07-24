"""
model.py - CNN 기반 주가 예측 모델 정의

이 파일은 캔들차트 이미지를 분석하여 주가의 상승/하락을 예측하는 CNN 모델들을 정의합니다.
- CNN5d: 5일 윈도우 데이터용 (32x15 이미지)
- CNN20d: 20일 윈도우 데이터용 (64x60 이미지)  
- CNN60d: 60일 윈도우 데이터용 (96x180 이미지) - 논문 추가

핵심 아이디어:
하루를 3픽셀로 표현 → 시가|고저가봉|종가
5일 = 15픽셀, 20일 = 60픽셀, 60일 = 180픽셀 너비의 캔들차트 이미지
"""
from __init__ import *


class CNN5d(nn.Module):
    """
    5일 윈도우 캔들차트 이미지 분석용 CNN 모델
    
    입력: [배치크기, 1, 32, 15] - 32픽셀 높이 × 15픽셀 너비 (5일×3픽셀)
    출력: [배치크기, 2] - 상승(1) 또는 하락(0) 확률
    
    구조: 2개의 합성곱 블록 + 완전연결층
    """
    
    def init_weights(self, m):
        """
        가중치 초기화: Xavier 초기화 방법 사용
        - 합성곱층과 완전연결층의 가중치를 적절히 초기화
        - 편향(bias)은 작은 양수값(0.01)으로 설정하여 dead neuron 방지
        """
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    
    def __init__(self):
        super(CNN5d, self).__init__()
        
        # === 첫 번째 합성곱 블록: 기본 특징 추출 ===
        self.conv1 = nn.Sequential(OrderedDict([
            ('Conv', nn.Conv2d(1, 64, (5, 3), padding=(2, 1), stride=(1, 1), dilation=(1, 1))), # [N, 64, 32, 15]
            ('BN', nn.BatchNorm2d(64, affine=True)),    # 배치 정규화로 학습 안정성 향상
            ('LeakyReLU', nn.LeakyReLU()),                         # 비선형 활성화
            ('Max-Pool', nn.MaxPool2d((2,1)))           # 세로 방향만 절반으로 압축: [N, 64, 16, 15]
        ]))
        self.conv1 = self.conv1.apply(self.init_weights)
        
        # === 두 번째 합성곱 블록: 고차원 특징 추출 ===
        self.conv2 = nn.Sequential(OrderedDict([
            ('Conv', nn.Conv2d(64, 128, (5, 3), padding=(2, 1), stride=(1, 1), dilation=(1, 1))), # [N, 128, 16, 15]
            ('BN', nn.BatchNorm2d(128, affine=True)),
            ('LeakyReLU', nn.LeakyReLU()),
            ('Max-Pool', nn.MaxPool2d((2,1)))           # 세로 방향 재압축: [N, 128, 8, 15]
        ]))
        self.conv2 = self.conv2.apply(self.init_weights)

        # === 분류층: 특징을 최종 예측으로 변환 ===
        self.DropOut = nn.Dropout(p=0.5)      # 50% 드롭아웃으로 과적합 방지
        self.FC = nn.Linear(15360, 2)          # 128×8×15 = 15,360 → 2개 클래스
        self.init_weights(self.FC)
        self.Softmax = nn.Softmax(dim=1)       # 확률값으로 변환 (합이 1)

    def forward(self, x):
        """
        순전파 과정: 캔들차트 이미지 → 상승/하락 확률
        
        Args:
            x: 입력 이미지 [배치크기, 32, 15]
            
        Returns:
            예측 확률 [배치크기, 2] - [하락확률, 상승확률]
        """
        # 채널 차원 추가: [N, 32, 15] → [N, 1, 32, 15] (흑백 이미지)
        x = x.unsqueeze(1).to(torch.float32)
        
        # 특징 추출 파이프라인
        x = self.conv1(x)    # [N, 64, 16, 15] - 기본 패턴 인식
        x = self.conv2(x)    # [N, 128, 8, 15] - 복합 패턴 인식
        
        # 분류 과정
        x = self.DropOut(x.view(x.shape[0], -1))  # 1차원으로 펼치고 드롭아웃 적용
        x = self.FC(x)                            # [N, 2] 최종 예측값
        x = self.Softmax(x)                       # 확률로 변환
        
        return x
    
    
    
class CNN20d(nn.Module):
    """
    20일 윈도우 캔들차트 이미지 분석용 CNN 모델
    
    입력: [배치크기, 1, 64, 60] - 64픽셀 높이 × 60픽셀 너비 (20일×3픽셀)
    출력: [배치크기, 2] - 상승(1) 또는 하락(0) 확률
    
    구조: 3개의 합성곱 블록 + 완전연결층
    - 더 큰 이미지를 처리하므로 CNN5d보다 하나의 합성곱 블록 추가
    - 더 깊은 네트워크로 장기간 패턴의 복잡한 특징 추출 가능
    """
    
    def init_weights(self, m):
        """
        가중치 초기화: CNN5d와 동일한 Xavier 초기화 방법
        """
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    
    def __init__(self):
        super(CNN20d, self).__init__()
        
        # === 첫 번째 합성곱 블록: 거친 패턴 감지 ===
        # stride=(3,1)과 dilation=(2,1)로 넓은 영역을 한번에 보면서 특징 추출
        self.conv1 = nn.Sequential(OrderedDict([
            ('Conv', nn.Conv2d(1, 64, (5, 3), padding=(3, 1), stride=(3, 1), dilation=(2, 1))), # [N, 64, 21, 60]
            ('BN', nn.BatchNorm2d(64, affine=True)),    # 배치 정규화
            ('LeakyReLU', nn.LeakyReLU()),                         # 활성화 함수
            ('Max-Pool', nn.MaxPool2d((2,1)))           # 세로 압축: [N, 64, 10, 60]
        ]))
        self.conv1 = self.conv1.apply(self.init_weights)
        
        # === 두 번째 합성곱 블록: 중간 단계 특징 추출 ===
        self.conv2 = nn.Sequential(OrderedDict([
            ('Conv', nn.Conv2d(64, 128, (5, 3), padding=(3, 1), stride=(1, 1), dilation=(1, 1))), # [N, 128, 12, 60]
            ('BN', nn.BatchNorm2d(128, affine=True)),
            ('LeakyReLU', nn.LeakyReLU()),
            ('Max-Pool', nn.MaxPool2d((2,1)))           # 세로 압축: [N, 128, 6, 60]
        ]))
        self.conv2 = self.conv2.apply(self.init_weights)
        
        # === 세 번째 합성곱 블록: 세밀한 고차원 특징 추출 ===
        self.conv3 = nn.Sequential(OrderedDict([
            ('Conv', nn.Conv2d(128, 256, (5, 3), padding=(2, 1), stride=(1, 1), dilation=(1, 1))), # [N, 256, 6, 60]
            ('BN', nn.BatchNorm2d(256, affine=True)),
            ('LeakyReLU', nn.LeakyReLU()),
            ('Max-Pool', nn.MaxPool2d((2,1)))           # 최종 압축: [N, 256, 3, 60]
        ]))
        self.conv3 = self.conv3.apply(self.init_weights)

        # === 분류층: 추출된 특징을 최종 예측으로 변환 ===
        self.DropOut = nn.Dropout(p=0.5)      # 과적합 방지
        self.FC = nn.Linear(46080, 2)          # 256×3×60 = 46,080 → 2개 클래스
        self.init_weights(self.FC)
        self.Softmax = nn.Softmax(dim=1)       # 확률 변환

    def forward(self, x):
        """
        순전파 과정: 20일 캔들차트 이미지 → 상승/하락 확률
        
        Args:
            x: 입력 이미지 [배치크기, 64, 60]
            
        Returns:
            예측 확률 [배치크기, 2] - [하락확률, 상승확률]
        """
        # 채널 차원 추가: [N, 64, 60] → [N, 1, 64, 60] (흑백 이미지)
        x = x.unsqueeze(1).to(torch.float32)
        
        # 3단계 특징 추출 파이프라인 - 점진적으로 추상화 레벨 증가
        x = self.conv1(x)    # [N, 64, 10, 60]  - 기본 가격 패턴 감지
        x = self.conv2(x)    # [N, 128, 6, 60]  - 중간 수준 트렌드 패턴 감지 
        x = self.conv3(x)    # [N, 256, 3, 60]  - 고차원 복합 패턴 감지
        
        # 최종 분류
        x = self.DropOut(x.view(x.shape[0], -1))  # 1차원으로 펼치고 드롭아웃
        x = self.FC(x)                            # [N, 2] 최종 분류
        x = self.Softmax(x)                       # 확률 변환
        
        return x


class CNN60d(nn.Module):
    """
    60일 윈도우 캔들차트 이미지 분석용 CNN 모델
    
    입력: [배치크기, 1, 96, 180] - 96픽셀 높이 × 180픽셀 너비 (60일×3픽셀)
    출력: [배치크기, 2] - 상승(1) 또는 하락(0) 확률
    
    구조: 4개의 합성곱 블록 + 완전연결층
    - 가장 큰 이미지를 처리하므로 4개의 합성곱 블록 사용
    - 장기간 패턴의 매우 복잡한 특징 추출 가능
    """
    
    def init_weights(self, m):
        """
        가중치 초기화: Xavier 초기화 방법
        """
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    
    def __init__(self):
        super(CNN60d, self).__init__()
        
        # === 첫 번째 합성곱 블록: 거친 패턴 감지 ===
        self.conv1 = nn.Sequential(OrderedDict([
            ('Conv', nn.Conv2d(1, 64, (5, 3), padding=(2, 1), stride=(3, 1), dilation=(3, 1))), # [N, 64, 32, 180]
            ('BN', nn.BatchNorm2d(64, affine=True)),
            ('LeakyReLU', nn.LeakyReLU()),
            ('Max-Pool', nn.MaxPool2d((2,1)))  # [N, 64, 16, 180]
        ]))
        self.conv1 = self.conv1.apply(self.init_weights)
        
        # === 두 번째 합성곱 블록: 중간 단계 특징 추출 ===
        self.conv2 = nn.Sequential(OrderedDict([
            ('Conv', nn.Conv2d(64, 128, (5, 3), padding=(2, 1), stride=(1, 1), dilation=(1, 1))), # [N, 128, 16, 180]
            ('BN', nn.BatchNorm2d(128, affine=True)),
            ('LeakyReLU', nn.LeakyReLU()),
            ('Max-Pool', nn.MaxPool2d((2,1)))  # [N, 128, 8, 180]
        ]))
        self.conv2 = self.conv2.apply(self.init_weights)
        
        # === 세 번째 합성곱 블록: 고차원 특징 추출 ===
        self.conv3 = nn.Sequential(OrderedDict([
            ('Conv', nn.Conv2d(128, 256, (5, 3), padding=(2, 1), stride=(1, 1), dilation=(1, 1))), # [N, 256, 8, 180]
            ('BN', nn.BatchNorm2d(256, affine=True)),
            ('LeakyReLU', nn.LeakyReLU()),
            ('Max-Pool', nn.MaxPool2d((2,1)))  # [N, 256, 4, 180]
        ]))
        self.conv3 = self.conv3.apply(self.init_weights)
        
        # === 네 번째 합성곱 블록: 최고차원 복합 패턴 추출 ===
        self.conv4 = nn.Sequential(OrderedDict([
            ('Conv', nn.Conv2d(256, 512, (5, 3), padding=(2, 1), stride=(1, 1), dilation=(1, 1))), # [N, 512, 4, 180]
            ('BN', nn.BatchNorm2d(512, affine=True)),
            ('LeakyReLU', nn.LeakyReLU()),
            ('Max-Pool', nn.MaxPool2d((2,1)))  # [N, 512, 2, 180]
        ]))
        self.conv4 = self.conv4.apply(self.init_weights)

        # === 분류층: 추출된 특징을 최종 예측으로 변환 ===
        self.DropOut = nn.Dropout(p=0.5)      # 과적합 방지
        self.FC = nn.Linear(184320, 2)         # 512×2×180 = 184,320 → 2개 클래스
        self.init_weights(self.FC)
        self.Softmax = nn.Softmax(dim=1)       # 확률 변환

    def forward(self, x):
        """
        순전파 과정: 60일 캔들차트 이미지 → 상승/하락 확률
        
        Args:
            x: 입력 이미지 [배치크기, 96, 180]
            
        Returns:
            예측 확률 [배치크기, 2] - [하락확률, 상승확률]
        """
        # 채널 차원 추가: [N, 96, 180] → [N, 1, 96, 180] (흑백 이미지)
        x = x.unsqueeze(1).to(torch.float32)
        
        # 4단계 특징 추출 파이프라인 - 점진적으로 추상화 레벨 증가
        x = self.conv1(x)    # [N, 64, 16, 180]   - 기본 가격 패턴 감지
        x = self.conv2(x)    # [N, 128, 8, 180]   - 중간 수준 트렌드 패턴 감지 
        x = self.conv3(x)    # [N, 256, 4, 180]   - 고차원 복합 패턴 감지
        x = self.conv4(x)    # [N, 512, 2, 180]   - 최고차원 장기 패턴 감지
        
        # 최종 분류
        x = self.DropOut(x.view(x.shape[0], -1))  # 1차원으로 펼치고 드롭아웃
        x = self.FC(x)                            # [N, 2] 최종 분류
        x = self.Softmax(x)                       # 확률 변환
        
        return x
    