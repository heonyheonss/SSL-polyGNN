from setuptools import setup, find_packages

# requirements.txt 파일의 내용을 읽어와 install_requires에 사용
with open('requirements.txt', encoding='utf-8') as f:
    required = f.read().splitlines()
    # 주석이나 특정 옵션이 포함된 라인은 제외할 수 있습니다.
    # 여기서는 간단하게 모든 라인을 포함시켰습니다.
    # torch 관련 라이브러리는 환경에 따라 설치 방법이 다르므로
    # install_requires에서 제외하고 사용자가 직접 설치하도록 유도하는 것이 더 안정적일 수 있습니다.
    # 예: required = [lib for lib in required if not lib.startswith('torch')]

setup(
    name='ssl_polygnn',  # 패키지 이름
    version='0.1.0',  # 패키지 버전
    author='Your Name',  # 작성자 이름
    author_email='your_email@example.com',  # 작성자 이메일
    description='A Self-Supervised Learning based Poly-GNN model for predicting multiple polymer properties.',
    long_description=open('README.md').read(), # 프로젝트 설명 (README.md 파일 필요)
    long_description_content_type='text/markdown',
    url='https://github.com/your_username/ssl-polygnn',  # Github Repository 주소
    packages=find_packages(exclude=['tests*', 'notebooks*']),  # 'tests'나 'notebooks' 폴더를 제외한 모든 패키지를 포함
    install_requires=required, # requirements.txt에서 읽어온 의존성 목록
    python_requires='>=3.8', # 요구되는 파이썬 버전
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Chemistry',
    ],
)
