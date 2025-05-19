import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# 제목 설정
st.title('코로나19 데이터 대시보드')

# 가상의 데이터 (실제 프로젝트에서는 API나 CSV에서 로드)
@st.cache_data  # 데이터 캐싱
def load_data():
    regions = ['서울', '경기', '부산', '대구', '기타']
    data = []
    for day in range(100):
        date = pd.Timestamp('2023-01-01') + pd.Timedelta(days=day)
        for region in regions:
            region_factor = {
                '서울': 1.5, '경기': 1.3, '부산': 0.8,
                '대구': 0.7, '기타': 0.5
            }.get(region, 1.0)
            new_cases = int((100 + day * 5 + day**1.5) * region_factor)
            data.append({
                '날짜': date,
                '지역': region,
                '신규확진자': new_cases
            })
    df = pd.DataFrame(data)
    df.sort_values(by=['지역', '날짜'], inplace=True)
    df['누적확진자'] = df.groupby('지역')['신규확진자'].cumsum()
    return df

data = load_data()

# 사이드바 - 필터링 옵션
st.sidebar.header('필터 옵션')
selected_regions = st.sidebar.multiselect(
    '지역 선택',
    options=data['지역'].unique(),
    default=data['지역'].unique()
)

# 날짜 범위 선택 - 에러 방지를 위한 처리 추가
min_date = data['날짜'].min().date()
max_date = data['날짜'].max().date()

try:
    date_range = st.sidebar.date_input(
        "날짜 범위 선택",
        [min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )
    
    # 날짜 범위가 완전히 선택되었는지 확인
    if len(date_range) < 2:
        st.sidebar.warning("시작일과 종료일을 모두 선택해주세요.")
        start_date = min_date
        end_date = max_date
    else:
        start_date = date_range[0]
        end_date = date_range[1]
except Exception as e:
    st.sidebar.warning(f"날짜 선택 중 오류가 발생했습니다: {e}")
    start_date = min_date
    end_date = max_date

st.sidebar.info(f"선택된 날짜 범위: {start_date} ~ {end_date}")

# 날짜 필터링 적용 함수 - pandas 최신 방식 사용
def filter_by_date(df, start, end):
    """날짜 필터링을 적용하는 함수 - 최신 pandas 권장 방식 사용"""
    # pd.to_datetime을 사용해 날짜 객체를 명시적으로 변환
    # np.array 사용으로 deprecated 경고 방지
    dates = np.array(pd.to_datetime(df['날짜']).dt.date)
    mask = (dates >= start) & (dates <= end)
    return df.loc[mask]

# 필터링된 데이터
filtered_data = data[data['지역'].isin(selected_regions)]
filtered_data = filter_by_date(filtered_data, start_date, end_date)

# 데이터저널리즘 기사 스타일 소개 추가
st.header('코로나19 추이로 보는 한국의 방역 정책 효과')
st.markdown("""
*'데이터로 읽는 코로나19' 시리즈 | 김데이터 기자*

지난 100일간 국내 코로나19 확진자 추이를 분석한 결과, 수도권과 비수도권의 확진자 발생 패턴에 뚜렷한 차이가 나타났다. 
서울과 경기 지역은 높은 인구 밀도에도 불구하고 사회적 거리두기 강화 이후 주간 확진자 증가폭이 크게 둔화되었다. 
특히, 서울은 1월 중순부터 시행된 강화된 방역조치로 인해 2월 첫째 주부터 신규 확진자가 감소하는 추세를 보였다.

반면 부산과 대구 지역은 방역수칙 완화 이후 산발적 집단감염이 발생하며 2월 중순부터 
확진자 수가 증가세로 전환되었다. 전문가들은 "지역별 방역정책의 시행 시기와 강도에 따라 
확진자 발생 패턴이 달라지고 있다"며 "지역 특성을 고려한 맞춤형 방역 전략이 필요하다"고 지적한다.

이번 분석은 중앙방역대책본부와 각 지자체가 제공한 일일 확진자 데이터를 지역별, 시간별로 집계하여 
패턴을 시각화한 것으로, 방역정책의 효과를 데이터로 검증할 수 있는 토대를 마련했다. 
아래 대시보드를 통해 지역별 확진자 추이와 시기별 방역정책 변화의 상관관계를 확인할 수 있다.
""")

# 주요 지표 표시
st.header('주요 지표')
col1, col2 = st.columns(2)

with col1:
    latest_date = filtered_data['날짜'].max()
    latest_cases = filtered_data[filtered_data['날짜'] == latest_date]['신규확진자'].sum()
    
    # 전날 데이터 계산
    prev_date = latest_date - pd.Timedelta(days=1)
    prev_cases = filtered_data[filtered_data['날짜'] == prev_date]['신규확진자'].sum() if prev_date in filtered_data['날짜'].values else 0
    
    # 전일 대비 증감 표시
    delta = latest_cases - prev_cases
    st.metric("최근 신규 확진자", f"{latest_cases:,}명", f"{delta:+,}")

with col2:
    # 최신 날짜의 모든 지역 누적확진자 합계
    total_cases = filtered_data[filtered_data['날짜'] == latest_date]['누적확진자'].sum()
    st.metric("누적 확진자", f"{total_cases:,}명")

# 시간에 따른 확진자 추이
st.header('시간에 따른 확진자 추이')
daily_data = filtered_data.groupby('날짜').sum().reset_index()
fig = px.line(daily_data, x='날짜', y='신규확진자', title='일별 신규 확진자 추이')

# 방역 정책 시기 강조 (회색 박스로 영역 표시)
highlight_periods = [
    {"start": "2023-01-10", "end": "2023-01-25", "label": "거리두기 강화", "color": "LightSalmon"},
    {"start": "2023-02-05", "end": "2023-02-20", "label": "방역 완화", "color": "LightGreen"},
]

for period in highlight_periods:
    fig.add_vrect(
        x0=period["start"], x1=period["end"],
        fillcolor=period["color"], opacity=0.3,
        layer="below", line_width=0,
        annotation_text=period["label"], annotation_position="top left"
    )

st.plotly_chart(fig)

# 지역별 확진자 현황
st.header('지역별 확진자 현황')
# 날짜 컬럼을 제외하고 그룹화하여 합산
region_data = filtered_data.groupby('지역').agg({'신규확진자': 'sum'}).reset_index()
fig = px.bar(region_data, x='지역', y='신규확진자', title='지역별 확진자 수')
st.plotly_chart(fig)

# 원본 데이터 표시
if st.checkbox('원본 데이터 보기'):
    st.subheader('필터링된 원본 데이터')
    st.dataframe(filtered_data)