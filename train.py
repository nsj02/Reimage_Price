"""
train.py - CNN 모델 훈련 및 성능 시각화 모듈

이 파일은 캔들차트 이미지로 주가 예측 모델을 훈련시키는 핵심 기능들을 포함합니다:
- 모델 훈련 및 검증 루프
- 조기 종료 (Early Stopping) 기능
- 훈련 과정 시각화 (손실/정확도 그래프)
"""
from __init__ import *

# GPU 사용 가능 시 CUDA, 없으면 CPU 사용
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_n_epochs(n_epochs, model, label_type, train_loader, valid_loader, criterion, optimizer, savefile, early_stop_epoch):
    """
    CNN 모델을 지정된 에포크 동안 훈련시키는 메인 함수
    
    주요 기능:
    1. 🏋️ 매 에포크마다 훈련 데이터로 모델 학습
    2. 📊 검증 데이터로 모델 성능 평가  
    3. 💾 성능이 향상될 때마다 최적 모델 저장
    4. ⏰ 성능 향상이 멈추면 조기 종료 (과적합 방지)
    
    Args:
        n_epochs (int): 최대 훈련 에포크 수
        model: 훈련할 CNN 모델 (CNN5d, CNN20d, 또는 CNN60d)
        label_type (str): 예측 라벨 타입 ('RET5', 'RET20', 또는 'RET60')
        train_loader: 훈련 데이터 로더
        valid_loader: 검증 데이터 로더  
        criterion: 손실 함수 (CrossEntropyLoss)
        optimizer: 최적화 알고리즘 (예: Adam)
        savefile (str): 모델 저장 경로
        early_stop_epoch (int): 조기 종료 기준 에포크 수
    
    Returns:
        tuple: (훈련손실, 검증손실, 훈련정확도, 검증정확도) 히스토리
    """
    # === 성능 추적 변수들 초기화 ===
    valid_loss_min = np.inf        # 최고 성능(최소 검증 손실) 추적
    train_loss_set = []            # 에포크별 훈련 손실 기록
    valid_loss_set = []            # 에포크별 검증 손실 기록
    train_acc_set = []             # 에포크별 훈련 정확도 기록
    valid_acc_set = []             # 에포크별 검증 정확도 기록
    invariant_epochs = 0           # 성능 향상이 없는 에포크 카운터 (조기 종료용)
    
    # === 메인 훈련 루프: 지정된 에포크만큼 반복 ===
    for epoch_i in range(1, n_epochs+1):
        
        # 현재 에포크의 손실/정확도 초기화
        train_loss, train_acc = 0.0, 0.0
        valid_loss, valid_acc = 0.0, 0.0
        
        # ============== 🏋️ 훈련 단계 (Training Phase) ==============
        model.train()  # 모델을 훈련 모드로 설정 (Dropout, BatchNorm 활성화)
        
        for i, (data, ret5, ret20, ret60, _, _, _) in enumerate(train_loader):
            # 라벨 타입에 따라 예측 타겟 선택
            assert label_type in ['RET5', 'RET20', 'RET60'], f"잘못된 라벨 타입: {label_type}"
            if label_type == 'RET5':
                target = ret5
            elif label_type == 'RET20':
                target = ret20
            else:  # RET60
                target = ret60

            # CrossEntropyLoss용 라벨 (원-핫 인코딩 불필요)
            target = target.long()

            # GPU로 데이터 이동
            data, target = data.to(device), target.to(device)
            
            # 경사도 초기화 (이전 배치의 경사도 제거)
            optimizer.zero_grad()
            
            # 순전파: 모델에 입력을 넣어 예측값 계산
            output = model(data)
            
            # 손실 계산
            loss = criterion(output, target)
            
            # 역전파: 손실에 대한 경사도 계산
            loss.backward()
            
            # 경사도를 이용한 가중치 업데이트
            optimizer.step()
            
            # 성능 지표 누적
            train_loss += loss.item() * data.size(0)  # 가중 평균을 위해 배치 크기 곱함
            train_acc += (output.argmax(1) == target).sum()  # 예측과 실제 비교

        # ============== 📊 검증 단계 (Validation Phase) ==============
        model.eval()  # 모델을 평가 모드로 설정 (Dropout 비활성화, BatchNorm 고정)
        
        with torch.no_grad():  # 검증 단계에서는 경사도 계산 비활성화 (메모리 절약)
            for i, (data, ret5, ret20, ret60, _, _, _) in enumerate(valid_loader):
                # 훈련 단계와 동일한 라벨 처리
                assert label_type in ['RET5', 'RET20', 'RET60'], f"잘못된 라벨 타입: {label_type}"
                if label_type == 'RET5':
                    target = ret5
                elif label_type == 'RET20':
                    target = ret20
                else:  # RET60
                    target = ret60
                    
                # CrossEntropyLoss용 라벨 (원-핫 인코딩 불필요)
                target = target.long()
                    
                # GPU로 데이터 이동
                data, target = data.to(device), target.to(device)
                
                # 순전파만 수행 (역전파 없음)
                output = model(data)
                loss = criterion(output, target)
                
                # 검증 성능 지표 누적
                valid_loss += loss.item() * data.size(0)
                valid_acc += (output.argmax(1) == target).sum()
        
        # ============== 📈 에포크별 성능 계산 및 기록 ==============
        
        # 평균 손실 계산 (전체 샘플 수로 나누기)
        train_loss = train_loss / len(train_loader.dataset)
        train_loss_set.append(train_loss)
        valid_loss = valid_loss / len(valid_loader.dataset)
        valid_loss_set.append(valid_loss)

        # 평균 정확도 계산 및 기록
        train_acc = train_acc / len(train_loader.dataset)
        train_acc_set.append(train_acc.detach().cpu().numpy())
        valid_acc = valid_acc / len(valid_loader.dataset)
        valid_acc_set.append(valid_acc.detach().cpu().numpy())
            
        # 현재 에포크 성능 출력
        print('Epoch: {} Training Loss: {:.6f} Validation Loss: {:.6f} Training Acc: {:.5f} Validation Acc: {:.5f}'.format(
            epoch_i, train_loss, valid_loss, train_acc, valid_acc))
        
        # ============== 💾 모델 저장 및 조기 종료 로직 ==============
        
        # 검증 손실이 개선되면 모델 저장
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
            valid_loss_min = valid_loss
            invariant_epochs = 0  # 개선되었으므로 카운터 리셋
            
            # 모델, 에포크, 옵티마이저 상태 저장
            torch.save({
                'epoch': epoch_i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, savefile)
        else:
            # 성능이 개선되지 않으면 카운터 증가
            invariant_epochs += 1
        
        # 조기 종료 조건 체크
        if invariant_epochs >= early_stop_epoch:
            print(f"Early Stop at Epoch [{epoch_i}]: Performance hasn't enhanced for {early_stop_epoch} epochs")
            break

    # 훈련 과정의 모든 성능 기록 반환
    return train_loss_set, valid_loss_set, train_acc_set, valid_acc_set



def plot_loss_and_acc(loss_and_acc_dict):
    """
    훈련 과정의 손실 및 정확도 변화를 시각화하는 함수
    
    📊 두 개의 그래프를 나란히 생성:
    - 왼쪽: 에포크별 손실(Loss) 변화 추이
    - 오른쪽: 에포크별 정확도(Accuracy) 변화 추이
    
    Args:
        loss_and_acc_dict (dict): 각 데이터셋별 [손실리스트, 정확도리스트] 딕셔너리
                                  예: {'train': ([손실들], [정확도들]), 'valid': ([손실들], [정확도들])}
    
    주요 기능:
    - 훈련/검증 성능을 한눈에 비교 가능
    - 과적합 여부 판단 (검증 손실이 훈련 손실보다 높아지는 지점)
    - 학습 수렴 상태 확인
    """
    # 2개 서브플롯 생성 (1행 2열, 가로로 배치)
    _, axes = plt.subplots(1, 2, figsize=(20, 6))
    
    # 첫 번째 데이터셋의 손실 리스트 길이로 총 에포크 수 계산
    tmp = list(loss_and_acc_dict.values())
    maxEpoch = len(tmp[0][0])

    # ========== 왼쪽 그래프: 손실(Loss) 시각화 ==========
    
    # Y축 범위 자동 설정 (모든 데이터셋의 최대/최소 손실 기준)
    maxLoss = max([max(x[0]) for x in loss_and_acc_dict.values()]) + 0.1
    minLoss = max(0, min([min(x[0]) for x in loss_and_acc_dict.values()]) - 0.1)

    # 각 데이터셋(훈련/검증)별로 손실 그래프 그리기
    for name, lossAndAcc in loss_and_acc_dict.items():
        axes[0].plot(range(1, 1 + maxEpoch), lossAndAcc[0], '-s', label=name)

    # 손실 그래프 설정
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_xticks(range(0, maxEpoch + 1, max(1, maxEpoch//10)))  # X축 눈금 (최소 간격 1)
    axes[0].axis([0, maxEpoch, minLoss, maxLoss])
    axes[0].legend()
    axes[0].set_title("Training & Validation Loss")

    # ========== 오른쪽 그래프: 정확도(Accuracy) 시각화 ==========
    
    # Y축 범위 자동 설정 (정확도는 0~1 사이이므로 범위 제한)
    maxAcc = min(1, max([max(x[1]) for x in loss_and_acc_dict.values()]) + 0.1)
    minAcc = max(0, min([min(x[1]) for x in loss_and_acc_dict.values()]) - 0.1)

    # 각 데이터셋별로 정확도 그래프 그리기  
    for name, lossAndAcc in loss_and_acc_dict.items():
        axes[1].plot(range(1, 1 + maxEpoch), lossAndAcc[1], '-s', label=name)

    # 정확도 그래프 설정
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_xticks(range(0, maxEpoch + 1, max(1, maxEpoch//10)))  # X축 눈금
    axes[1].axis([0, maxEpoch, minAcc, maxAcc])
    axes[1].legend()
    axes[1].set_title("Training & Validation Accuracy")
    
    # 그래프 표시
    plt.tight_layout()  # 서브플롯 간격 자동 조정
    plt.show()