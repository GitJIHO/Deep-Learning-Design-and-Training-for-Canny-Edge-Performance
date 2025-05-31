# Canny Edge 결과를 위한 딥러닝 모델 설계 및 학습

## 개요

### 프로젝트 목적
본 프로젝트는 입력 이미지로부터 Canny Edge 검출 결과를 생성하는 딥러닝 모델을 설계하고 학습하는 것을 목표로 합니다. 기존의 BSDS500 데이터셋의 Ground Truth 대신 Canny Edge 알고리즘을 통해 새로운 대체 GT를 구축하여 모델을 학습시키고, 성능을 평가합니다.

### 주요 기능
- **데이터셋 처리**: BSDS500 데이터셋을 활용한 이미지 전처리 및 Ground Truth 생성
- **모델 설계**: HED(Holistically-Nested Edge Detection) 기반의 개선된 딥러닝 모델
- **학습 시스템**: 다중 스케일 손실 함수를 활용한 효율적 학습
- **성능 평가**: 새로운 GT 및 기존 GT와의 정확도 비교 분석

## 이론적 배경

### Canny Edge Detection
Canny Edge Detection은 1986년 John F. Canny가 개발한 엣지 검출 알고리즘으로, 다음과 같은 단계를 거칩니다:
1. **노이즈 제거**: 가우시안 블러를 통한 노이즈 감소
2. **그래디언트 계산**: Sobel 필터를 사용한 강도 그래디언트 계산
3. **비최대 억제**: 엣지가 아닌 픽셀 제거
4. **이중 임계값**: 높은/낮은 임계값을 통한 엣지 분류
5. **연결성 분석**: 약한 엣지의 연결 여부 판단

### HED (Holistically-Nested Edge Detection)
HED는 VGG-16을 백본으로 하는 완전 합성곱 신경망으로, 다중 스케일에서 엣지를 검출합니다:
- **다중 사이드 출력**: 각 합성곱 블록에서 엣지 예측
- **심층 감독**: 모든 사이드 출력에 대한 손실 계산
- **융합 네트워크**: 다중 스케일 특징의 통합

## 구현 코드 설명

### 1. 데이터셋 관리 (`FixedHEDBSDSManager`)
```python
class FixedHEDBSDSManager:
    def __init__(self):
        self.drive_path = '/content/drive/MyDrive/Deep Learning/HED-BSDS.tar'
        self.local_path = './HED-BSDS.tar'
```
- Google Drive에서 데이터셋 자동 다운로드 및 압축 해제
- BSDS500 표준 구조로 데이터 재구성
- 이미지 크기 통일 (320×480) 및 전처리

### 2. 개선된 Ground Truth 생성
```python
def generate_improved_canny_gt(self, image_path, target_size=(320, 480)):
    # 노이즈 제거 및 전처리
    denoised = cv2.medianBlur(gray, 5)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    
    # 적응적 임계값 설정
    high_thresh = np.percentile(blurred, 85)
    low_thresh = high_thresh * 0.4
    
    # 다중 스케일 Canny 적용
    canny1 = cv2.Canny(blurred, low_thresh, high_thresh)
    canny2 = cv2.Canny(blurred, low_thresh*0.8, high_thresh*0.8)
```

### 3. 개선된 HED 모델 (`ImprovedHEDModel`)
```python
class ImprovedHEDModel(nn.Module):
    def __init__(self):
        super(ImprovedHEDModel, self).__init__()
        # 5개의 합성곱 블록
        self.conv1 = nn.Sequential(...)  # 64 channels
        self.conv2 = nn.Sequential(...)  # 128 channels
        self.conv3 = nn.Sequential(...)  # 256 channels
        self.conv4 = nn.Sequential(...)  # 512 channels
        self.conv5 = nn.Sequential(...)  # 512 channels
        
        # 사이드 출력 층들
        self.side1-5 = nn.Sequential(...)
        
        # 융합 층
        self.fuse = nn.Sequential(...)
```

주요 개선사항:
- **Batch Normalization**: 학습 안정성 향상
- **Dropout**: 과적합 방지 (0.1~0.3)
- **Xavier 초기화**: 가중치 초기화 최적화

## 코드 분석

### 주요 함수 및 매개변수

#### 1. 손실 함수 (`ImprovedEdgeLoss`)
```python
class ImprovedEdgeLoss(nn.Module):
    def __init__(self):
        super(ImprovedEdgeLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(3.0))
```
- **pos_weight=3.0**: 엣지 픽셀의 가중치를 3배로 설정하여 클래스 불균형 해결
- **가중 융합**: 각 사이드 출력에 서로 다른 가중치 적용 `[0.5, 0.75, 1.0, 0.75, 0.5, 1.0]`

#### 2. 최적화 설정
```python
optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
```
- **AdamW**: Adam의 개선된 버전으로 가중치 감쇠 적용
- **학습률**: 1e-5로 설정하여 안정적 학습
- **스케줄러**: 검증 손실이 개선되지 않으면 학습률 감소

#### 3. 데이터 증강
```python
# 수평 뒤집기
if self.use_augmentation and np.random.random() > 0.5:
    image = np.fliplr(image).copy()

# 밝기/대비 조정
brightness = np.random.uniform(0.8, 1.2)
contrast = np.random.uniform(0.8, 1.2)
```

### 평가 메트릭
```python
def calculate_metrics(pred, gt, threshold=0.5):
    # Accuracy, Precision, Recall, F1-Score 계산
    TP = (pred_bin * gt_bin).sum().item()
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8)
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
```

## 실험 결과

### 학습 설정
- **에포크**: 30회
- **배치 크기**: 4
- **이미지 크기**: 320×480
- **학습 데이터**: 50개 이미지
- **테스트 데이터**: 30개 이미지

### 성능 평가 결과
본 프로젝트에서는 두 가지 관점에서 모델 성능을 평가했습니다:

1. **새로운 GT (Canny Edge)와 모델 결과 비교**
   - 모델이 학습한 대상과의 일치도 측정
   - 높은 정확도가 예상됨

2. **기존 GT와 모델 결과 비교**
   - 원본 BSDS500 Ground Truth와의 비교
   - 실제 사람이 라벨링한 엣지와의 차이 분석

### 학습 곡선 분석
- **Training Loss**: 에포크에 따른 훈련 손실 감소 추이
- **Validation Loss**: 검증 손실을 통한 과적합 여부 확인
- **조기 종료**: 검증 손실이 개선되지 않을 때 학습 중단

## 개선 가능성

### 1. 모델 아키텍처 개선
- **ResNet/DenseNet 백본**: VGG-16 대신 더 깊은 네트워크 활용
- **Attention 메커니즘**: 중요한 엣지 영역에 집중
- **Multi-Scale Training**: 다양한 해상도에서 학습

### 2. 데이터 증강 확대
```python
# 추가 가능한 증강 기법
- 회전 변환 (rotation)
- 스케일 변경 (scaling)
- 노이즈 추가 (noise injection)
- 색상 지터링 (color jittering)
```

### 3. 손실 함수 개선
- **Focal Loss**: 어려운 샘플에 더 집중
- **Dice Loss**: 세그멘테이션 성능 향상
- **Perceptual Loss**: 인식적 품질 개선

### 4. 후처리 기법
```python
# 모폴로지 연산을 통한 엣지 정제
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
refined_edges = cv2.morphologyEx(prediction, cv2.MORPH_CLOSE, kernel)
```

### 5. 앙상블 방법
- 여러 모델의 예측 결과 결합
- TTA (Test Time Augmentation) 적용

## 결론

### 실험 결과
#### === 1) 새로운 GT(Canny Edge)와 모델 결과간 Accuracy 측정 ===

Accuracy: 0.8812 (88.12%)

#### === 2) 기존 GT와 모델 결과간 Accuracy 측정 ===
Accuracy: 0.5003 (50.03%)

![image](https://github.com/user-attachments/assets/36f515c2-4590-4afa-83c6-7adabaab1cc1)

![image](https://github.com/user-attachments/assets/dd475982-9a83-4b0f-90d1-fb592a307860)

#### 📊 사용자 이미지 평가 결과:
   Accuracy: 0.9737 (97.37%)

### 프로젝트 성과
본 프로젝트에서는 Canny Edge Detection 알고리즘을 Ground Truth로 활용하여 딥러닝 모델을 성공적으로 학습시켰습니다. 주요 성과는 다음과 같습니다:

1. **데이터셋 구축**: BSDS500을 기반으로 Canny Edge GT 생성
2. **모델 설계**: HED 기반의 개선된 엣지 검출 모델 구현
3. **학습 시스템**: 안정적이고 효율적인 학습 파이프라인 구축
4. **성능 평가**: 정량적 메트릭을 통한 객관적 평가

### 학습 내용
- **엣지 검출의 이해**: 전통적 방법과 딥러닝 방법의 차이점
- **다중 스케일 학습**: 여러 해상도에서의 특징 추출 및 융합
- **불균형 데이터 처리**: 엣지/비엣지 픽셀 비율 불균형 해결
- **전이 학습**: 사전 훈련된 가중치 활용 방법

### 실용적 의의
이 모델은 다음과 같은 분야에 활용 가능합니다:
- **컴퓨터 비전**: 객체 검출, 세그멘테이션의 전처리
- **의료 영상**: X-ray, MRI 이미지의 경계 검출
- **자율 주행**: 도로 경계 및 차선 검출
- **산업 검사**: 제품 결함 검출 및 품질 관리

본 프로젝트를 통해 전통적인 컴퓨터 비전 알고리즘과 현대 딥러닝 기법을 성공적으로 결합하여, 실용적이고 효과적인 엣지 검출 시스템을 구축할 수 있었습니다.
