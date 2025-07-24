"""
utils.py - 유틸리티 함수 모음

이 파일은 프로젝트 전반에서 사용되는 공통 유틸리티 함수들을 모아둔 파일입니다:
1. ⏱️ timer: 함수 실행 시간 측정 데코레이터
2. 🖼️ display_image: 캔들차트 이미지 시각화 함수  
3. 🔧 Dict2ObjParser: YAML 설정을 객체로 변환하는 클래스
"""
from __init__ import *


@contextmanager 
def timer(name: str, _align): 
    """
    ⏱️ 함수 실행 시간을 측정하고 출력하는 컨텍스트 매니저 데코레이터
    
    사용법:
        @timer('데이터 로드', '10')
        def load_data():
            # 시간이 오래 걸리는 작업
            pass
    
    Args:
        name (str): 작업 이름 (출력에 표시될 이름)
        _align (str): 출력 시 정렬을 위한 너비값
        
    출력 예시:
        [ 데이터 로드 ] | 2023-04-27 15:30:45 Done | Using 12.345 seconds
    """
    s = time.time()                           # 시작 시간 기록
    yield                                     # 여기서 실제 함수 실행
    elapsed = time.time() - s                 # 실행 시간 계산
    print(f"{ '[' + name + ']' :{_align}} | {time.strftime('%Y-%m-%d %H:%M:%S')} Done | Using {elapsed: .3f} seconds")
    

def display_image(entry):
    """
    🖼️ 캔들차트 이미지와 라벨을 시각화하는 함수
    
    dataset.py에서 생성된 이미지 데이터를 matplotlib으로 출력하여
    캔들차트가 올바르게 생성되었는지 육안으로 확인할 수 있습니다.
    
    Args:
        entry (list): [이미지배열, ret5_라벨, ret20_라벨] 형태의 데이터
                     - 이미지배열: numpy 2D 배열 (높이 x 너비)
                     - ret5_라벨: 5일 후 수익률 라벨 (0: 하락, 1: 상승)
                     - ret20_라벨: 20일 후 수익률 라벨 (0: 하락, 1: 상승)
    
    출력:
        흑백 캔들차트 이미지 + 라벨 정보가 포함된 matplotlib 창
    """
    # 입력 데이터 형태 검증
    assert (type(entry) == list) and (len(entry) == 3), "Type error, expected a list with length of 3"
    
    plt.figure                                        # 새 그래프 창 생성
    plt.imshow(entry[0], cmap=plt.get_cmap('gray'))  # 흑백 이미지로 표시
    plt.ylim((0,entry[0].shape[0]-1))                # Y축 범위 설정 (이미지 높이)
    plt.xlim((0,entry[0].shape[1]-1))                # X축 범위 설정 (이미지 너비)
    plt.title(f'ret5: {entry[1]}\nret20: {entry[2]}')  # 라벨 정보를 제목으로 표시
    

class Dict2ObjParser():
    """
    🔧 딕셔너리(또는 YAML)를 객체로 변환하는 클래스
    
    YAML 설정 파일을 읽으면 중첩된 딕셔너리 형태가 되는데,
    이를 점(.) 표기법으로 접근할 수 있는 객체로 변환합니다.
    
    변환 전: setting['TRAIN']['BATCH_SIZE']  (딕셔너리 접근)
    변환 후: setting.TRAIN.BATCH_SIZE        (객체 속성 접근)
    
    사용 예시:
        with open('config.yml', 'r') as f:
            config_dict = yaml.safe_load(f)
        
        parser = Dict2ObjParser(config_dict)
        setting = parser.parse()
        
        print(setting.TRAIN.BATCH_SIZE)  # 64
        print(setting.MODEL)             # 'CNN5d'
    """
    
    def __init__(self, nested_dict):
        """
        초기화 함수
        
        Args:
            nested_dict (dict): 변환할 중첩된 딕셔너리
        """
        self.nested_dict = nested_dict

    def parse(self):
        """
        딕셔너리를 객체로 변환하는 메인 함수
        
        Returns:
            namedtuple: 점 표기법으로 접근 가능한 객체
        
        Raises:
            TypeError: 입력이 딕셔너리가 아닌 경우
        """
        nested_dict = self.nested_dict
        if (obj_type := type(nested_dict)) is not dict:
            raise TypeError(f"Expected 'dict' but found '{obj_type}'")
        return self._transform_to_named_tuples("root", nested_dict)

    def _transform_to_named_tuples(self, tuple_name, possibly_nested_obj):
        """
        재귀적으로 딕셔너리를 namedtuple로 변환하는 내부 함수
        
        작동 원리:
        1. 딕셔너리 → namedtuple로 변환 (키들이 속성명이 됨)
        2. 리스트 → 각 원소를 재귀적으로 변환
        3. 기본 타입 (문자열, 숫자 등) → 그대로 반환
        
        Args:
            tuple_name (str): 생성할 namedtuple의 이름
            possibly_nested_obj: 변환할 객체 (딕셔너리, 리스트, 기본타입)
            
        Returns:
            변환된 객체 (namedtuple, list, 또는 기본타입)
        """
        if type(possibly_nested_obj) is dict:
            # 딕셔너리인 경우: namedtuple로 변환
            named_tuple_def = namedtuple(tuple_name, possibly_nested_obj.keys())
            transformed_value = named_tuple_def(
                *[
                    self._transform_to_named_tuples(key, value)  # 각 값도 재귀적으로 변환
                    for key, value in possibly_nested_obj.items()
                ]
            )
        elif type(possibly_nested_obj) is list:
            # 리스트인 경우: 각 원소를 재귀적으로 변환
            transformed_value = [
                self._transform_to_named_tuples(f"{tuple_name}_{i}", possibly_nested_obj[i])
                for i in range(len(possibly_nested_obj))
            ]
        else:
            # 기본 타입인 경우: 그대로 반환
            transformed_value = possibly_nested_obj

        return transformed_value

