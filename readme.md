# Hand Bone Image Segmentation
X-ray 이미지에서 사람의 뼈를 Segmentation 하는 인공지능 만들기
## CV 5팀 소개
> ### 멤버
<table>
    <tr height="160px">
        <td align="center" width="150px">
            <a href="https://github.com/woohee-yang"><img height="120px" width="120px" src="https://github.com/boostcampaitech6/level2-objectdetection-cv-05/assets/78292486/a1e74529-0abf-4d80-9716-4e8ae5ec8e72"/></a>
            <br/>
            <a href="https://github.com/woohee-yang"><strong>양우희</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/jinida"><img height="120px" width="120px" src="https://github.com/boostcampaitech6/level2-objectdetection-cv-05/assets/78292486/28955c1d-fa4e-46b1-9d70-f98eb54109b2"/></a>
            <br />
            <a href="https://github.com/jinida"><strong>이영진</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/cmj5064"><img height="120px" width="120px" src="https://github.com/boostcampaitech6/level2-objectdetection-cv-05/assets/78292486/6388976d-d0bd-4ba6-bae8-6c7e6c5b3352"></a>
            <br/>
            <a href="https://github.com/cmj5064"><strong>조민지</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/ccsum19"><img height="120px" width="120px" src="https://github.com/boostcampaitech6/level2-objectdetection-cv-05/assets/78292486/9ad5ecc3-e5be-4738-99c2-cc6e7f3931cb"/></a>
            <br/>
            <a href="https://github.com/ccsum19"><strong>조수민</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/hee000"><img height="120px" width="120px" src="https://github.com/boostcampaitech6/level2-objectdetection-cv-05/assets/78292486/cde48fcd-8099-472b-9877-b2644954ec68"/></a>
            <br />
            <a href="https://github.com/hee000"><strong>조창희</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
              <a href="https://github.com/SangBeom-Hahn"><img height="120px" width="120px" src="https://github.com/boostcampaitech6/level2-objectdetection-cv-05/assets/78292486/1f7ed5a5-5e0f-46e4-85c6-31b9767dce41"/></a>
              <br />
              <a href="https://github.com/SangBeom-Hahn"><strong>한상범</strong></a>
              <br />
          </td>
    </tr>
</table>
<br/>

## Hand Bone Semantic Segmentation 프로젝트 
> ### 대회 개요
- 본 프로젝트는 X-ray 이미지에서 사람 뼈를 segmentation하는 모델을 완성하는 것을 목표로 한다.  
- 모델은 각 클래스(29개)에 대한 확률 맵을 갖는 멀티채널 예측을 수행하고, 이를 기반으로 각 픽셀을 해당 클래스에 할당한다.
- hand bone x-ray 객체가 담긴 이미지가 모델의 인풋으로 사용하며, segmentation annotation은 json file로 제공한다. 
<br/>

> ### 팀 역할
|이름|역할|
|------|---|
|전체|좋은 성능의 모델을 완성할 수 있도록 노력하기|
|조민지|Augmentation 실험|
|조수민|Baseline 코드 작성, argparser 작성, 검증 셋 실험, augmentation 실험, 모델 실험, 모델 앙상블 실험 |
|조창희|모델 실험 |
<br/>


> ### 개발환경
```bash
- Language : Python
- Environment
  - CPU : Intel(R) Xeon(R) Gold 5120
  - GPU : Tesla V100-SXM2 32GB × 1
- Framework : PyTorch
- Collaborative Tool : Git, Notion
```
<br/>

> ### Dataset
- 전체 이미지 개수 : 1088장 (train 800장, test 288장)
- 사람 별로 두 장의 이미지 (왼손, 오른손)
- 총 29개의 클래스
```python
CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]
```
- 손가락, 손등, 팔로 구성
- 반지 낀 손가락 등 outlier 존재

> ### Training
```bash
python train.py 
```
train 과정에서 BCE + DICE loss 사용
<br/>

> ### Inference
```bash
python inference.py 
```
<br/>

> ### Ensemble
- 결과 CSV 파일들 soft voting을 위한 ensemble 코드
- 예측 결과가 과반수 이상일 때 true

> ### 📂 File Tree
```
📦 level2-cv-semanticsegmentation-cv-05
├── __init__.py
├── dataset.py
├── ensemble.py
├── inference.py
├── model.py
├── requirements.txt
├── readme.md
└── train.py

```
