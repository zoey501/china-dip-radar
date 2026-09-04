from datetime import date, timedelta, datetime
from zoneinfo import ZoneInfo
import time
import numpy as np
import pandas as pd
import streamlit as st
import akshare as ak

TRADING_DAYS = 252
CN_TZ = ZoneInfo("Asia/Shanghai")
st.set_page_config(page_title="A股/ETF 抄底雷达 V4.2", layout="wide")

POPULAR_STOCKS = {
    "601899":"紫金矿业","600519":"贵州茅台","300750":"宁德时代","002594":"比亚迪",
    "601318":"中国平安","600036":"招商银行","601088":"中国神华","600900":"长江电力",
    "600031":"三一重工","601138":"工业富联","000858":"五粮液",
    "002371":"北方华创","300308":"中际旭创","688981":"中芯国际",
}
POPULAR_ETFS = {
    "510300":"沪深300ETF","510050":"上证50ETF","510500":"中证500ETF",
    "588000":"科创50ETF","159915":"创业板ETF","512480":"半导体ETF",
    "512660":"军工ETF","512800":"银行ETF","515790":"光伏ETF",
    "516160":"新能源ETF","159928":"消费ETF","518880":"黄金ETF",
}

def normalize_hist(df):
    mp={"日期":"Date","开盘":"Open","收盘":"Close","最高":"High","最低":"Low","成交量":"Volume"}
    x=df.rename(columns=mp).copy()
    x["Date"]=pd.to_datetime(x["Date"])
    for c in ["Open","High","Low","Close","Volume"]:
        if c in x.columns:
            x[c]=pd.to_numeric(x[c],errors="coerce")
    return x.sort_values("Date").set_index("Date")

def retry(fn, attempts=3):
    err=None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            err=e
            time.sleep(0.8*(i+1))
    raise err

@st.cache_data(ttl=1800, show_spinner=False)
def load_hist(code, asset, years=4):
    start=(date.today()-timedelta(days=365*years+40)).strftime("%Y%m%d")
    end=date.today().strftime("%Y%m%d")
    def fetch():
        if asset=="ETF":
            return ak.fund_etf_hist_em(symbol=code,period="daily",start_date=start,end_date=end,adjust="qfq")
        return ak.stock_zh_a_hist(symbol=code,period="daily",start_date=start,end_date=end,adjust="qfq")
    df=retry(fetch)
    if df is None or df.empty:
        raise ValueError("没有取得历史数据")
    return normalize_hist(df)

def features(x):
    x=x.copy()
    x["ret"]=x.Close.pct_change()
    x["rv20"]=x.ret.rolling(20).std()*np.sqrt(TRADING_DAYS)
    x["sigma_prev"]=x.rv20.shift(1)/np.sqrt(TRADING_DAYS)
    x["down_step"]=(-x.ret/x.sigma_prev).replace([np.inf,-np.inf],np.nan)
    x["ma60"]=x.Close.rolling(60).mean()
    x["ma250"]=x.Close.rolling(250).mean()
    d=x.Close.diff()
    gain=d.clip(lower=0).rolling(14).mean()
    loss=(-d.clip(upper=0)).rolling(14).mean()
    rs=gain/loss.replace(0,np.nan)
    x["rsi"]=100-100/(1+rs)
    for n in [5,20,60]:
        x[f"fwd{n}"]=x.Close.shift(-n)/x.Close-1
    return x

def rank_last(s,n=750):
    s=pd.Series(s).dropna().iloc[-n:]
    if len(s)<20:return np.nan
    return float((s<=s.iloc[-1]).mean()*100)

def analyze(code,name,asset,years=4):
    h=features(load_hist(code,asset,years))
    if len(h)<80:
        return None
    last=h.iloc[-1]
    price=float(last.Close)
    prev=float(h.Close.iloc[-2])
    rv=float(last.rv20) if pd.notna(last.rv20) and last.rv20>0 else 0.25
    step=-(price/prev-1)/(rv/np.sqrt(TRADING_DAYS))
    dd60=price/float(h.Close.iloc[-60:].max())-1
    dev60=price/float(h.Close.iloc[-60:].mean())-1
    rsi=float(last.rsi) if pd.notna(last.rsi) else 50
    rvr=rank_last(h.rv20)

    score=0
    if step>=2:score+=32
    elif step>=1.5:score+=26
    elif step>=1:score+=18
    elif step>=.5:score+=9
    if dd60<=-.20:score+=22
    elif dd60<=-.15:score+=18
    elif dd60<=-.10:score+=13
    elif dd60<=-.06:score+=7
    if dev60<=-.15:score+=15
    elif dev60<=-.10:score+=11
    elif dev60<=-.06:score+=7
    elif dev60<=-.03:score+=3
    if rsi<=20:score+=12
    elif rsi<=25:score+=9
    elif rsi<=30:score+=6
    elif rsi<=35:score+=3
    if np.isfinite(rvr):
        if rvr>=95:score+=12
        elif rvr>=90:score+=9
        elif rvr>=80:score+=6
        elif rvr>=70:score+=3
    score=int(np.clip(score,0,100))
    state=("等待" if score<30 else "观察" if score<50 else "第一档"
           if score<65 else "较强抄底" if score<80 else "极端恐慌")
    return {
        "代码":code,"名称":name,"类型":asset,
        "最新价":price,"涨跌%":(price/prev-1)*100,"步距σ":step,
        "60日回撤%":dd60*100,"RSI14":rsi,"RV20%":rv*100,"RV分位%":rvr,
        "抄底分":score,"状态":state,"_hist":h
    }

def backtest(h):
    rows=[]
    for step in [.5,1,1.5,2,2.5,3]:
        sig=h[h.down_step>=step]
        for hold in [5,20,60]:
            r=sig[f"fwd{hold}"].dropna()
            rows.append({
                "步距":step,"持有日":hold,"次数":len(r),
                "平均收益":r.mean() if len(r) else np.nan,
                "胜率":(r>0).mean() if len(r) else np.nan,
                "最差":r.min() if len(r) else np.nan
            })
    return pd.DataFrame(rows)

st.title("🇨🇳 A股 / ETF 抄底雷达 V4.2")
st.success("页面已成功加载。V4.2 不会在打开网页时自动请求行情；只有点击按钮才开始读取数据。")
st.caption("热门榜｜自选扫描｜单票查询。优先稳定性，使用最新日线数据。")

tab1,tab2,tab3=st.tabs(["🔥 热门抄底榜","⭐ 自选扫描","🔎 单票查询"])

with tab1:
    st.subheader("热门抄底排名")
    kind=st.radio("选择榜单",["热门A股","热门ETF","混合榜"],horizontal=True)
    st.write("点击下面按钮后才会开始计算。第一次可能需要一些时间。")
    if st.button("▶ 生成热门抄底榜",type="primary"):
        universe=[]
        if kind in ["热门A股","混合榜"]:
            universe += [(c,n,"A股") for c,n in POPULAR_STOCKS.items()]
        if kind in ["热门ETF","混合榜"]:
            universe += [(c,n,"ETF") for c,n in POPULAR_ETFS.items()]

        rows=[]
        bar=st.progress(0)
        status=st.empty()
        total=len(universe)
        for i,(code,name,asset) in enumerate(universe,1):
            status.write(f"正在分析 {name} ({code})… {i}/{total}")
            try:
                r=analyze(code,name,asset,years=4)
                if r:
                    r.pop("_hist",None)
                    rows.append(r)
            except Exception:
                pass
            bar.progress(i/total)
        status.empty()

        if rows:
            df=pd.DataFrame(rows).sort_values(["抄底分","步距σ"],ascending=False)
            st.session_state["hot_v42"]=df
        else:
            st.error("本次没有成功取得榜单数据。")

    if "hot_v42" in st.session_state:
        st.dataframe(st.session_state["hot_v42"],use_container_width=True,hide_index=True)

with tab2:
    st.subheader("扫描自己的自选")
    text=st.text_area("代码用逗号分隔","601899,600519,300750,600031,510300,588000")
    st.caption("ETF 常见代码以 51 / 15 / 56 / 58 开头，会自动识别。")
    if st.button("▶ 扫描自选股",type="primary"):
        codes=[x.strip() for x in text.replace("，",",").split(",") if x.strip()]
        rows=[]
        bar=st.progress(0)
        for i,code in enumerate(codes,1):
            asset="ETF" if code.startswith(("51","15","56","58")) else "A股"
            try:
                r=analyze(code,code,asset,years=4)
                if r:
                    r.pop("_hist",None)
                    rows.append(r)
            except Exception:
                pass
            bar.progress(i/len(codes))
        if rows:
            st.dataframe(pd.DataFrame(rows).sort_values("抄底分",ascending=False),use_container_width=True,hide_index=True)
        else:
            st.error("没有成功读取这些代码。")

with tab3:
    st.subheader("单票查询")
    c1,c2=st.columns([2,1])
    code=c1.text_input("股票 / ETF代码","601899")
    asset=c2.selectbox("类型",["A股","ETF"])
    if st.button("▶ 查询这只股票",type="primary"):
        with st.spinner("正在读取历史行情并计算…"):
            try:
                r=analyze(code.strip(),code.strip(),asset,years=11)
                if not r:
                    st.error("数据不足")
                else:
                    h=r["_hist"]
                    a,b,c,d,e=st.columns(5)
                    a.metric("最新价",f"{r['最新价']:.3f}",f"{r['涨跌%']:.2f}%")
                    b.metric("抄底分",f"{r['抄底分']}/100",r["状态"])
                    c.metric("步距",f"{r['步距σ']:.2f}σ")
                    d.metric("60日回撤",f"{r['60日回撤%']:.1f}%")
                    e.metric("RSI14",f"{r['RSI14']:.1f}")

                    prev=float(h.Close.iloc[-2])
                    daily=prev*(r["RV20%"]/100)/np.sqrt(TRADING_DAYS)
                    ladder=[]
                    for s in [.5,1,1.5,2,2.5,3]:
                        px=prev-s*daily
                        ladder.append({"步距":s,"触发价":px,"已触发":"✅" if r["最新价"]<=px else ""})
                    st.subheader("分档买入价")
                    st.dataframe(pd.DataFrame(ladder).style.format({"触发价":"{:.3f}"}),use_container_width=True)

                    st.subheader("约10年历史回测")
                    bt=backtest(h)
                    bt2=bt.copy()
                    for col in ["平均收益","胜率","最差"]:
                        bt2[col]=bt2[col].map(lambda v:f"{v:.2%}" if pd.notna(v) else "")
                    st.dataframe(bt2,use_container_width=True)

                    st.subheader("长期位置")
                    st.line_chart(pd.DataFrame({"收盘":h.Close,"MA60":h.ma60,"MA250":h.ma250}))
            except Exception as e:
                st.error(f"读取失败：{e}")

st.divider()
st.caption(f"北京时间：{datetime.now(CN_TZ):%Y-%m-%d %H:%M:%S} ｜ V4.2 使用按需加载，打开页面本身不会请求大量行情。")
