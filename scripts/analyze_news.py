import os
import sys
import logging
import json
from dotenv import load_dotenv
import httpx
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
import psycopg2
from psycopg2 import pool


# |------------------------------------------------------|
# |--- 1. 초기 설정: 로깅, 환경 변수, LLM, DB 커넥션 풀 ---|
# |------------------------------------------------------|

# 로거(Logger) 설정: print() 대신 logging을 사용하여 체계적인 로그를 남기기
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# .env 파일에서 환경 변수 로드
load_dotenv()

# 필수 환경 변수 목록
REQUIRED_ENV_VARS = ["OPENAI_API_KEY", "DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD"]

# 환경 변수 유효성 검사
missing_vars = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
if missing_vars:
    logging.error(f"필수 환경 변수가 .env 파일에 설정되지 않았습니다: {', '.join(missing_vars)}")
    sys.exit(1)

# SSL 검증 비활성화를 위한 커스텀 HTTP 클라이언트 생성
custom_http_client = httpx.Client(verify=False)

# LangChain의 OpenAI 클라이언트 설정... 
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, http_client=custom_http_client, api_key=os.getenv("OPENAI_API_KEY"))   
#slm = ChatOpenAI(model="gpt-4.1-nano", temperature=0, http_client=custom_http_client, api_key=os.getenv("OPENAI_API_KEY")) #nano 모델 실험

# 데이터베이스 연결 정보
db_conn_info = {"host": os.getenv("DB_HOST"),"database": os.getenv("DB_NAME"),"user": os.getenv("DB_USER"),"password": os.getenv("DB_PASSWORD")}

# DB 커넥션 풀(Connection Pool) 초기화
try:
    db_pool = psycopg2.pool.SimpleConnectionPool(1, 5, **db_conn_info)
    logging.info("데이터베이스 커넥션 풀이 성공적으로 생성되었습니다.")
except psycopg2.OperationalError as e:
    logging.error(f"데이터베이스 연결에 실패했습니다: {e}")
    sys.exit(1)

# |---------------------------------------------|
# |-- 2. Few-Shot 예시 및 마스터 프롬프트 정의 ---|
# |---------------------------------------------|

# 퓨샷 프롬프트는 실제 기사문들 원문 전체 복붙해 넣기
FEW_SHOT_EXAMPLES = {
    # --- Corn Examples ---
    "Corn": [
        {
            "article": 
            """
            Wheat surges by daily limit, corn sharply higher after Russia invades Ukraine
            Reuters News
            25 Feb 2022 02:47:35
            CHICAGO, Feb 24 (Reuters)
            * Wheat and soybeans hit highest since 2012
            * Corn jumps to eight-month peak
            * Black Sea grain supplies uncertain after Russia invades Ukraine
            (Rewrites throughout, adds quote, updates prices, changes byline, adds dateline)
            By Karl Plume
            CHICAGO, Feb 24 (Reuters) - U.S. wheat futures spiked by their daily trading limit on Thursday to their highest since mid-2012 and corn futures rallied to eight-month peaks after Russian forces attacked Ukraine, exacerbating worries over global grain supplies.
            Soybean futures eased on profit-taking after setting fresh 9-1/2 year highs overnight on concerns about global vegetable oil supplies amid conflict in the major sunflower oil producing region.
            Russian forces invaded Ukraine by land, air and sea, confirming the worst fears of the West with the biggest attack by one state on another in Europe since World War Two. [nL1N2UZ089]
            Russia and Ukraine account for about 29% of global wheat exports, 19% of corn supplies and 80% of sunflower oil exports. Traders worry the conflict could trigger a scramble to replace those supplies. [nL1N2UX0MK]
            Ukraine's military suspended commercial shipping at its ports and Moscow suspended the movement of commercial vessels in the Azov Sea until further notice, though it kept Russian ports in the Black Sea open. [nL1N2UZ0L3]
            Still, top wheat importer Egypt cancelled its latest purchasing tender after receiving just one offer after the invasion. [nL1N2UZ1OC]
            "With the ports shut down, that takes a big chunk of grain off the global market and that might send more business to the U.S.," said Ted Seifried, chief agriculture strategist for the Zaner Group.
            Chicago Board of Trade May wheat was up its daily 50-cent trading limit at $9.34-3/4 a bushel by 11:20 a.m. CST (1620 GMT), the highest point for a most-active contract since July 2012.
            May corn was up 14-3/4 cents at $6.96 a bushel after earlier peaking at an eight-month high of $7.16-1/4.
            May soybeans topped at $17.59-1/4 a bushel, the highest for a most-active contract since September 2012, but retreated to $16.70-3/4, down 1/4 cent.
            All U.S. wheat, corn, soybean and soyoil contracts posted life-of-contract highs on Thursday.
            """,
            "output":{
                "sentiment_score":90,
                "reasoning":"Major supply shock as war jeopardizes ~19% of global corn supplies from the Black Sea region. While not limit up like wheat, the rally to an 8-month high reflects a severe, bullish disruption.",
                "keywords":["Russia-Ukraine War", "Supply Shock", "Black Sea Exports","Port Closure", "ExtremePositive", "Grain Supply Chain"]                
            }
        },
        {
            "article": 
            """
            US corn, soybeans extend losses as Midwest weather looks crop-friendly
            By P.J. Huffstutter
            Reuters News
            23 Jul 2025 04:59:22
            Soybeans see choppy trading session amid U.S.-China trade talk news
            Wheat futures rise despite global supply expectations
            Brazil's corn production forecast pressures U.S. corn market
            Adds closing U.S. prices, market moves, new headline, updates bullet
            CHICAGO, July 22 (Reuters) - Chicago Board of Trade corn futures extended session losses on Tuesday, weighed down by forecasts for crop-friendly rain in U.S. grain belts this week.
            Soybean futures closed modestly lower after a choppy session, turning higher at times on support from a decline in U.S. crop ratings and news that U.S. and Chinese trade officials will meet to discuss an August 12 deadline for a deal to avert sharply higher tariffs.
            Wheat futures firmed, despite expectations of plentiful global supplies, after Russia trimmed its crop harvest and export forecasts, market analysts said. The first new-crop wheat from Russia, the world's biggest wheat exporter, has arrived on the market, traders and analysts said.
            The most active CBOT wheat contract Wv1 ended the day 7-1/4 cents higher at $5.49-1/2 per bushel. The most active corn contract Cv1 settled down 4-1/4 cents at $4.18 a bushel, while soybeans Sv1 ended 1/2-cent lower at $10.25-1/2 a bushel.
            Traders had been concerned that high temperatures in the U.S. Midwest would damage yields, but after a brief burst of heat mid-week, showers are expected to limit crop stress, said Commodity Weather Group.
            This summer's spate of hot weather and frequent rains created a greenhouse-like effect, boosting corn prospects. The U.S. Department of Agriculture on Monday rated 74% of the U.S. corn crop, the highest assessment for corn at this time of year since 2016.
            "This is mostly a weather market we're seeing," said Jim Gerlach, president of U.S. brokerage A/C Trading. Gerlach added that traders may also be starting to factor in whether the pending August 1 tariff deadline for most U.S. trading partners will actually happen.
            U.S. Treasury Secretary Scott Bessent said on Tuesday that he will meet his Chinese counterpart next week in Stockholm and discuss what is likely to be an extension of an August 12 deadline for a deal to avert sharply higher tariffs.
            The corn market also continued to feel some pressure from Monday's report from agribusiness consultancy AgRural, which increased its estimate for Brazil's total 2024/25 corn production to 136.3 million tons, up from 130.6 million tons, market analysts said.
            """,
            "output":{
                "sentiment_score":20,
                "reasoning":"Price down as crop-friendly weather, record-high USDA crop ratings (74% good/excellent), and increased Brazil production estimates all point to a large supply.",
                "keywords":["Favorable Weather","High Crop Ratings", "Brazil Production", "Supply Pressure", "VeryNegative"]
            }
        },
        {
            "article": 
            """
            CBOT corn ends steady-firm on export demand; USDA monthly data awaited
            Reuters News
            11 Jul 2025 04:41:47
            CHICAGO, July 10 (Reuters) - Chicago Board of Trade corn futures ended mostly steady to firmer on Thursday on strong weekly U.S. export data and short-covering ahead of monthly supply/demand reports due on Friday, but near-ideal crop weather bolstered crop prospects, capping rallies, traders said.
            CBOT September corn CU25 settled unchanged at $3.99-1/4 per bushel after matching a life-of-contract low set a day earlier at $3.96-1/4.
            New-crop CBOT December corn CZ25 ended up 1 cent at $4.16-1/2 a bushel.
            The U.S. Department of Agriculture reported export sales of U.S. old-crop corn in the week ended July 3 at 1,262,100 metric tons, above a range of trade expectations for 375,000 to 900,000 tons. Sales of new-crop corn for the week were 888,600 tons, also above expectations. EXP/CORN
            Separately, under its daily reporting rules, the USDA confirmed private sales of 110,000 metric tons of U.S. new-crop corn to undisclosed destinations.
            Ahead of Friday's monthly USDA supply/demand reports, analysts surveyed by Reuters on average expected the government to lower its forecasts of U.S. corn inventories remaining at the end of the 2024/25 and 2025/26 marketing years.
            Brazilian government crop supply agency Conab raised its estimate of the country's total 2024/25 corn crop to 131.97 million metric tons, a new record that was up 2.9% from last month's estimate of 128.25 million tons.
            """,
            "output":{
                "sentiment_score":55,
                "reasoning":"A neutral outcome as strong, better-than-expected U.S. export data (Demand↑) was completely offset by near-ideal crop weather and a record Brazil crop forecast (Supply↑).",
                "keywords":["US Export Sales", "Crop Weather", "Brazil Record Crop", "USDA Report","Neutral","Mixed Signals"]
            }
        },
        {
            "article": 
            """
            U.S. corn sales to Mexico set record high
            NoticiasFinancieras - English
            21 Nov 2024 22:12:43
            Braulio Carbajal La Jornada Newspaper
            Thursday, November 21, 2024, p. 17
            The corn shipment business -mainly yellow and transgenic- from the United States to Mexico is experiencing its best moment by registering unprecedented levels, according to official data. This in a context in which both countries have been facing for months a panel of trade disputes over the limitation of imports of genetically modified grain for human consumption, which, as recently revealed by the Ministry of Economy, Mexico has lost.
            Data from the United States Department of Agriculture (USDA) reveal that in the first nine months of 2024, Mexico's corn purchases from that country amounted to 4.252 billion dollars, a record for the same period since records have been kept, and which is on track to break the historical mark of 5.386 billion dollars reported for the whole of 2023.
            The sharp increase in the value of Mexico's imports from the US comes at a time when production volume has plummeted 33 percent, given that between January and September 2024, according to the Agricultural Markets Consulting Group (GCMA), the country has only produced 9 million 300 thousand tons, while at the same time in 2023, the figure was 14 million tons.
            This, together with the drop in international grain prices, has caused a plunge in the value of Mexico's corn production, which from one year to the next went from 4,999 million dollars to 2,882 million dollars.
            Production in 2024 will be the lowest in the last 10 years and, according to GCMA estimates, in 2025 it will be even lower due to the effects of the drought and the lack of programs to promote increased productivity and profitability.
            On February 13, 2023, the Mexican government issued a presidential decree to prohibit the use of genetically modified grain in tortillas or dough, as well as the prohibition of the use of transgenic corn in all products for human consumption and animal feed, which led the US to convene a dispute panel in August of that year within the framework of the T-MEC.
            As informed last Wednesday by Marcelo Ebrard, head of the Ministry of Economy, Mexico was informed last October 22 -in a preliminary result- that it lost in the dispute panel; however, the official assured, Mexico replied again to the dispute panel, and it will be next December 14 when a final ruling will be issued.
            In recent years, Mexico has faced tensions regarding its position to prohibit the importation of transgenic corn for human consumption, especially white corn, a staple in the Mexican diet. The recent unfavorable decision in the trade panel requested by the US and Canada could force Mexico to modify its position or face possible trade retaliation, said Juan Carlos Anaya, general director of the GCMA.
            He pointed out that the lack of a clear regulatory framework to identify and label non-GM corn in imports and consumption generates uncertainty about its purity, making it difficult to guarantee that it is free of traces of GM corn. This situation allows mixing without strict quality and safety controls.
            For the specialist, there are three scenarios: elimination of the decree, which would avoid trade retaliation; imposition of tariffs by the US and Canada, which would be selective, on key Mexican products such as avocados, tequila or auto parts, which would reduce competitiveness and affec
            """,
            "output":{
                "sentiment_score":85,
                "reasoning":"Record-high U.S. sales to Mexico driven by a structural supply deficit, as Mexican production plummets 33%. Mexico's loss in a trade dispute over GM corn further solidifies strong, long-term U.S. export demand.",
                "keywords":["US-Mexico Trade", "Record Export Sales", "Mexico Production Deficit", "GM Corn Dispute", "Strong Demand","VeryPositive"]
            }
        },
    ],
    # --- Wheat Examples ---
    "Wheat": [
        {
            "article": 
            """
            Wheat surges by daily limit, corn sharply higher after Russia invades Ukraine
            Reuters News
            25 Feb 2022 02:47:35
            CHICAGO, Feb 24 (Reuters)
            * Wheat and soybeans hit highest since 2012
            * Corn jumps to eight-month peak
            * Black Sea grain supplies uncertain after Russia invades Ukraine
            (Rewrites throughout, adds quote, updates prices, changes byline, adds dateline)
            By Karl Plume
            CHICAGO, Feb 24 (Reuters) - U.S. wheat futures spiked by their daily trading limit on Thursday to their highest since mid-2012 and corn futures rallied to eight-month peaks after Russian forces attacked Ukraine, exacerbating worries over global grain supplies.
            Soybean futures eased on profit-taking after setting fresh 9-1/2 year highs overnight on concerns about global vegetable oil supplies amid conflict in the major sunflower oil producing region.
            Russian forces invaded Ukraine by land, air and sea, confirming the worst fears of the West with the biggest attack by one state on another in Europe since World War Two. [nL1N2UZ089]
            Russia and Ukraine account for about 29% of global wheat exports, 19% of corn supplies and 80% of sunflower oil exports. Traders worry the conflict could trigger a scramble to replace those supplies. [nL1N2UX0MK]
            Ukraine's military suspended commercial shipping at its ports and Moscow suspended the movement of commercial vessels in the Azov Sea until further notice, though it kept Russian ports in the Black Sea open. [nL1N2UZ0L3]
            Still, top wheat importer Egypt cancelled its latest purchasing tender after receiving just one offer after the invasion. [nL1N2UZ1OC]
            "With the ports shut down, that takes a big chunk of grain off the global market and that might send more business to the U.S.," said Ted Seifried, chief agriculture strategist for the Zaner Group.
            Chicago Board of Trade May wheat was up its daily 50-cent trading limit at $9.34-3/4 a bushel by 11:20 a.m. CST (1620 GMT), the highest point for a most-active contract since July 2012.
            May corn was up 14-3/4 cents at $6.96 a bushel after earlier peaking at an eight-month high of $7.16-1/4.
            May soybeans topped at $17.59-1/4 a bushel, the highest for a most-active contract since September 2012, but retreated to $16.70-3/4, down 1/4 cent.
            All U.S. wheat, corn, soybean and soyoil contracts posted life-of-contract highs on Thursday.
            """
            ,
            "output":{
                "sentiment_score":95,
                "reasoning":"Extreme supply shock as war removes ~29% of global wheat exports from the market. Port shutdowns confirmed. Unexpected invasion (Exp: Extreme) caused limit up (Mag: Extreme).",
                "keywords":["Russia-Ukraine War", "Supply Shock", "Black Sea Exports", "Port Closure", "LimitUp", "ExtremePositive", "Food Security"]
            }
        },
        {
            "article": 
            """
            US corn, soybeans extend losses as Midwest weather looks crop-friendly
            By P.J. Huffstutter
            Reuters News
            23 Jul 2025 04:59:22
            Soybeans see choppy trading session amid U.S.-China trade talk news
            Wheat futures rise despite global supply expectations
            Brazil's corn production forecast pressures U.S. corn market
            Adds closing U.S. prices, market moves, new headline, updates bullet
            CHICAGO, July 22 (Reuters) - Chicago Board of Trade corn futures extended session losses on Tuesday, weighed down by forecasts for crop-friendly rain in U.S. grain belts this week.
            Soybean futures closed modestly lower after a choppy session, turning higher at times on support from a decline in U.S. crop ratings and news that U.S. and Chinese trade officials will meet to discuss an August 12 deadline for a deal to avert sharply higher tariffs.
            Wheat futures firmed, despite expectations of plentiful global supplies, after Russia trimmed its crop harvest and export forecasts, market analysts said. The first new-crop wheat from Russia, the world's biggest wheat exporter, has arrived on the market, traders and analysts said.
            The most active CBOT wheat contract Wv1 ended the day 7-1/4 cents higher at $5.49-1/2 per bushel. The most active corn contract Cv1 settled down 4-1/4 cents at $4.18 a bushel, while soybeans Sv1 ended 1/2-cent lower at $10.25-1/2 a bushel.
            Traders had been concerned that high temperatures in the U.S. Midwest would damage yields, but after a brief burst of heat mid-week, showers are expected to limit crop stress, said Commodity Weather Group.
            This summer's spate of hot weather and frequent rains created a greenhouse-like effect, boosting corn prospects. The U.S. Department of Agriculture on Monday rated 74% of the U.S. corn crop, the highest assessment for corn at this time of year since 2016.
            "This is mostly a weather market we're seeing," said Jim Gerlach, president of U.S. brokerage A/C Trading. Gerlach added that traders may also be starting to factor in whether the pending August 1 tariff deadline for most U.S. trading partners will actually happen.
            U.S. Treasury Secretary Scott Bessent said on Tuesday that he will meet his Chinese counterpart next week in Stockholm and discuss what is likely to be an extension of an August 12 deadline for a deal to avert sharply higher tariffs.
            The corn market also continued to feel some pressure from Monday's report from agribusiness consultancy AgRural, which increased its estimate for Brazil's total 2024/25 corn production to 136.3 million tons, up from 130.6 million tons, market analysts said.
            """,
            "output":{
                "sentiment_score":65,
                "reasoning":"Price firmed despite ample global supply, driven by bullish news of Russia, a top exporter, trimming its harvest and export forecasts. (Supply↓)",
                "keywords":["Russia Export Forecast", "Supply Concerns", "Bullish", "VeryPositive"]
            }
        },
        {
            "article": 
            """
            CBOT wheat finishes lower on improving Russian crop outlook
            Reuters News
            04 Jul 2024 04:23:08
            CHICAGO, July 3 (Reuters) - Chicago Board of Trade wheat futures closed weaker on Wednesday on hopes that top-supplier Russia will produce a bigger crop than previously expected, traders said.
            Russian agricultural consultancy Sovecon said it had raised its 2024 wheat crop forecast to 84.1 million tonnes from 80.7 million.
            Russian wheat export prices have declined for four weeks in a row, tracking global markets on news of an abundant harvest.
            The U.S. Department of Agriculture is slated to issue weekly U.S. grain and soy export sales data on Friday, one day later than usual due to the U.S. Independence Day holiday on Thursday. Analysts expect 2024-25 U.S. wheat export sales of 350,000-700,000 metric tons for the week ended June 27.
            CBOT markets will be closed on Thursday.
            CBOT September soft red winter wheat WU24 closed down 7 cents at $5.74 a bushel.
            K.C. September hard red winter wheat KWU24 slid 8-3/4 cents to end at $5.83-1/2 a bushel, and MGEX September spring wheat MWEU24 last traded down 9-3/4 cents at $6.21-1/4 a bushel.
            European wheat prices also dipped.
            """,
            "output":{
                "sentiment_score":35,
                "reasoning":"Price lower as a Russian consultancy raised its wheat crop forecast for the world's top supplier, signaling a more abundant global supply than previously expected.",
                "keywords":["Russian Crop Forecast","Sovecon","Abundant Supply","Export Prices","MildlyNegative"]
            }
        },
        {
            "article": 
            """
            CBOT wheat falls on harvest pressure, favorable weather
            Reuters News
            25 Jun 2025 04:26:34
            CHICAGO, June 24 (Reuters) - Chicago Board of Trade wheat sank on Tuesday on pressure from the ongoing harvest in the U.S. Plains and Black Sea as well as a lack of weather threats, analysts said.
            CBOT September soft red winter wheat WU25 settled down 17-1/2 cents at $5.52 per bushel.
            K.C. September hard red winter wheat KWU25 fell 15-1/4 cents to end at $5.49-3/4 a bushel and Minneapolis July spring wheat MWEN25 finished down 3-1/2 cents at $6.41-3/4 a bushel.
            Dry weather in the U.S. Plains has helped the winter wheat crop dry down after drenching rains in previous weeks, and experts predict the wheat harvest will accelerate this week.
            Broadly favorable production prospects for wheat across the Northern Hemisphere were also weighing on the market.
            Argus Media has increased its forecast for Russia's 2025/26 wheat production, now projecting output to reach 84.8 million tons and come in well above last year's 81.3 million tons.
            The U.S. Department of Agriculture rated 49% of the U.S. winter wheat crop and 54% of the spring crop good to excellent as of Sunday in its weekly crop progress and condition report. Those ratings were both 3 points below the previous week and 3 points below the average analyst estimates in a Reuters poll.
            The winter wheat crop harvest accelerated to 19% complete, up from 10% last week.
            """,
            "output":{
                "sentiment_score":25,
                "reasoning":"Price sank under strong seasonal harvest pressure from the U.S. and Black Sea. Favorable weather and an increased Russian production forecast outweighed a bullishly lower-than-expected USDA crop rating.",
                "keywords":["Harvest Pressure", "Favorable Weather","Russian Production","USDA Crop Ratings","VeryNegative"]
            }
        },
        {
            "article": 
            """
            PARIS, July 25 (Reuters) - Ratings of French soft wheat crops showed 69% were in good or excellent condition by July 21, unchanged from a week earlier and up from 50% a year ago, farm office FranceAgriMer said on Friday. The harvest of soft wheat was 86% complete, moving ahead of the previous week's 75% progress.
            """,
            "output":{
                "sentiment_score": 45,
                "reasoning":"Good crop conditions in France (69% G/E) point to ample supply (Supply↑). Bearish, but impact on CBOT is moderate as per geographical weighting.",
                "keywords":["French Crop Condition", "Ample Supply", "Harvest Progress", "FranceAgriMer", "MildlyNegative"]
            }
        },
    ],
    "Soybean": [
        {
            "article": 
            """
            Wheat surges by daily limit, soyoil hits record high after Russia invades Ukraine
            Reuters News
            25 Feb 2022 03:10:17
            CHICAGO, Feb 24 (Reuters)
            * Black Sea grain supplies uncertain after invasion
            * Wheat highest since 2012, corn at eight-month peak
            * Soybeans slip from 9-1/2 year high on profit taking
            * Soyoil futures hit all-time high on vegoil supply concerns
            (New throughout, updates prices, market activity and comments, adds soyoil futures at record high)
            By Karl Plume
            CHICAGO, Feb 24 (Reuters) - U.S. wheat futures spiked by their daily trading limit on Thursday to their highest since mid-2012 and corn futures rallied to eight-month peaks after Russian forces attacked Ukraine, exacerbating worries over global grain supplies.
            Soyoil futures notched an all-time high on concerns about global vegetable oil supplies amid conflict in the major sunflower oil producing region. Soybean futures eased on profit-taking after setting fresh 9-1/2 year highs overnight.
            Russian forces invaded Ukraine by land, air and sea, confirming the worst fears of the West with the biggest attack by one state on another in Europe since World War Two. [nL1N2UZ089]
            Russia and Ukraine account for about 29% of global wheat exports, 19% of corn supplies and 80% of sunflower oil exports. Traders worry the conflict could trigger a scramble to replace those supplies. [nL1N2UX0MK]
            Ukraine's military suspended commercial shipping at its ports and Moscow suspended the movement of commercial vessels in the Azov Sea until further notice, though it kept Russian ports in the Black Sea open. [nL1N2UZ0L3]
            Still, top wheat importer Egypt cancelled its latest purchasing tender after receiving just one offer after the invasion. [nL1N2UZ1OC]
            "With the ports shut down, that takes a big chunk of grain off the global market and that might send more business to the U.S.," said Ted Seifried, chief agriculture strategist for the Zaner Group.
            Chicago Board of Trade May wheat was up its daily 50-cent trading limit at $9.34-3/4 a bushel by 11:58 a.m. CST (1658 GMT), the highest point for a most-active contract since July 2012.
            May corn was up 14-3/4 cents at $6.96 a bushel after earlier peaking at an eight-month high of $7.16-1/4.
            May soybeans topped at $17.59-1/4 a bushel, the highest for a most-active contract since September 2012, but retreated to $16.66-1/4, down 4-3/4 cents. May soyoil was up 2.45 cents at 73.03 cents per pound after peaking at 74.58 cents, the highest on record for a most-active contract .
            All U.S. wheat, corn, soybean and soyoil contracts posted life-of-contract highs on Thursday.
            (Additional reporting by Sybille de La Hamaide in Paris, Enrico Dela Cruz in Manila and Emily Chow in Beijing Editing by David Goodman, Mark Potter, Diane Craft and David Gregorio) ((karl.plume@thomsonreuters.com; +1 313 484 5285; Reuters Messaging: karl.plume.thomsonreuters.com@reuters.net))
            """
            ,
            "output":{
                "sentiment_score":45,
                "reasoning":"Price retreated and closed lower on the day due to profit-taking, despite the broader bullish macro context. The session's price action itself was negative.",
                "keywords":["Profit-Taking", "Price Retreat", "Short-Term Bearish", "Mixed Signals"]
            }
        },
        {
            "article":  
            """
            US corn, soybeans extend losses as Midwest weather looks crop-friendly
            By P.J. Huffstutter
            Reuters News
            23 Jul 2025 04:59:22
            Soybeans see choppy trading session amid U.S.-China trade talk news
            Wheat futures rise despite global supply expectations
            Brazil's corn production forecast pressures U.S. corn market
            Adds closing U.S. prices, market moves, new headline, updates bullet
            CHICAGO, July 22 (Reuters) - Chicago Board of Trade corn futures extended session losses on Tuesday, weighed down by forecasts for crop-friendly rain in U.S. grain belts this week.
            Soybean futures closed modestly lower after a choppy session, turning higher at times on support from a decline in U.S. crop ratings and news that U.S. and Chinese trade officials will meet to discuss an August 12 deadline for a deal to avert sharply higher tariffs.
            Wheat futures firmed, despite expectations of plentiful global supplies, after Russia trimmed its crop harvest and export forecasts, market analysts said. The first new-crop wheat from Russia, the world's biggest wheat exporter, has arrived on the market, traders and analysts said.
            The most active CBOT wheat contract Wv1 ended the day 7-1/4 cents higher at $5.49-1/2 per bushel. The most active corn contract Cv1 settled down 4-1/4 cents at $4.18 a bushel, while soybeans Sv1 ended 1/2-cent lower at $10.25-1/2 a bushel.
            Traders had been concerned that high temperatures in the U.S. Midwest would damage yields, but after a brief burst of heat mid-week, showers are expected to limit crop stress, said Commodity Weather Group.
            This summer's spate of hot weather and frequent rains created a greenhouse-like effect, boosting corn prospects. The U.S. Department of Agriculture on Monday rated 74% of the U.S. corn crop, the highest assessment for corn at this time of year since 2016.
            "This is mostly a weather market we're seeing," said Jim Gerlach, president of U.S. brokerage A/C Trading. Gerlach added that traders may also be starting to factor in whether the pending August 1 tariff deadline for most U.S. trading partners will actually happen.
            U.S. Treasury Secretary Scott Bessent said on Tuesday that he will meet his Chinese counterpart next week in Stockholm and discuss what is likely to be an extension of an August 12 deadline for a deal to avert sharply higher tariffs.
            The corn market also continued to feel some pressure from Monday's report from agribusiness consultancy AgRural, which increased its estimate for Brazil's total 2024/25 corn production to 136.3 million tons, up from 130.6 million tons, market analysts said.
            """,
            "output":{
                "sentiment_score":50,
                "reasoning":"A choppy session with the price closing nearly flat. Bullish U.S.-China trade talk news and lower crop ratings were offset by broader market pressure.",
                "keywords":["Mixed Signals","Choppy Trading","US-China Trade Talks","Crop Ratings","Neutral"]
            }
        },
        {
            "article": 
            """
            Soybeans head for 2% weekly gain on US demand hopes
            Reuters News
            18 Jul 2025 09:57:53
            CANBERRA, July 18 (Reuters) - Chicago soybean futures held steady on Friday and were on track for a weekly gain of about 2%, supported by hopes for improved U.S. exports and expectations that the U.S. biofuel policy will boost demand for soyoil used as feedstock.
            However, prices were still less than $1 a bushel above last year's five-year lows, with plentiful supply from South America and projections of a large U.S. harvest capping gains.
            Corn futures inched higher and were headed for a roughly 2% weekly gain due to a wave of bargain-hunting and short-covering earlier in the week, with plentiful supply also weighing on the market.
            Wheat climbed but was set for a weekly loss of about 2% amid seasonal pressure from ongoing northern hemisphere harvests.
            FUNDAMENTALS
            * The most-active soybean contract on the Chicago Board of Trade (CBOT) Sv1 was flat at $10.26-3/4 a bushel, as of 0033 GMT, while CBOT corn Cv1 was up 0.2% at $4.22 a bushel and wheat Wv1 was 0.4% higher at $5.35-3/4 a bushel.
            * CBOT soyoil BOc1 rose 0.1% and was headed for a roughly 3.5% weekly gain, having earlier in the week touched its highest level in nearly two years.
            * U.S. soybean export sales in the week ended July 10 reached 529,600 metric tons for 2025-26 shipment, the U.S. Department of Agriculture (USDA) said, beating analysts' expectations.
            * This week, the USDA also reported a sale of 120,000 tons of U.S. soybeans to "unknown destinations", triggering speculation that China might be the buyer and could buy again.
            * A trade deal between the United States and Indonesia could also boost U.S. soy exports.
            * However, oilseed lobby group Abiove raised its forecast for Brazil's 2024/25 soybean exports to 109 million tons from 108.2 million tons, and the Rosario Grains Exchange lifted its estimate for Argentina's 2024/25 harvest to 49.5 million tons from 48.5 million tons.
            * PepsiCo PEP.O, meanwhile, said it was expanding use of avocado and olive oil across its brands, rather than the canola or soybean oil it uses.
            """,
            "output":{
                "sentiment_score":60,
                "reasoning":"Weekly gain driven by strong, better-than-expected U.S. export sales and biofuel demand hopes. However, gains are capped by plentiful supply from South America and large U.S. harvest projections.",
                "keywords":["US Export Sales","Biofuel Demand", "South American Supply", "Harvest Projections", "Mixed Signals", "MildlyPositive"]
            }
        },
        {
            "article": 
            """
            Soy surges 2%; corn, wheat rally after Trump pauses most tariffs
            By Renee Hickman and Julie Ingwersen
            Reuters News
            10 Apr 2025 04:32:03
            Rewrites throughout with market rally after Trump pauses tariffs; adds closing U.S. prices
            CHICAGO, April 9 (Reuters) - U.S. soybean futures soared by 2% on Wednesday while corn and wheat closed higher, rallying along with crude oil CLc1 and equity markets after U.S. President Donald Trump said he would pause the tariff increases he announced last week for most countries, even as he raised them on China.
            On the Chicago Board of Trade, May soybeans SK25 settled up 20 cents at $10.12-3/4 per bushel. CBOT May corn CK25 ended up 5 cents at $4.74 a bushel and May wheat WK25 rose 2-1/4 cents to finish at $5.43-1/4 a bushel.
            After a choppy start to the session, all three grain markets surged after Trump's announcement.
            "It was instantaneous. Trump posted on social media and, just like that, kind of changed the narrative" on tariffs, said Terry Linn, analyst with Chicago-based Linn & Associates.
            Grain markets had been affected by a tit-for-tat tariff war that erupted last week between the United States and China, but soybeans were hit hardest, slumping to a four-month low on Monday.
            China is the world's biggest soy buyer and takes in around half of U.S. soybean exports each year. Beijing's willingness to retaliate against U.S. tariffs has led to fears of weaker demand for U.S. soybeans, said analysts.
            Yet CBOT soybeans led the gains on Tuesday, with soyoil futures BOv1 getting a lift from a jump in crude oil futures. Soyoil is the main U.S. feedstock for biodiesel fuel. O/R
            Firm U.S. cash markets for soybeans lent support as well. Farmers have been reluctant to book fresh sales of soybeans or corn at current prices, a factor that has tightened pipeline supplies and prompted some exporters and domestic processors to raise their bids in order to draw out fresh supplies.
            Also supporting Chicago soybeans - and corn - was the prospect of increased demand for U.S. biofuel, following a recommendation from an industry coalition to sharply raise federal mandates for biomass diesel blending in 2026.
            Wheat gained some support from a weaker U.S. dollar and from dry weather and hot temperatures expected in the U.S. Plains this weekend.
            Traders were also positioning themselves ahead of monthly supply/demand report due from the U.S. Department of Agriculture on Thursday, analysts said.
            """,
            "output":{
                "sentiment_score":75,
                "reasoning":"Price surged 2% on broad market relief after U.S. paused most tariff hikes. Strong crude oil, firm cash markets, and biofuel hopes provided further support, outweighing specific concerns over higher tariffs on China.",
                "keywords":["US Tariff Policy","Risk-On Sentiment", "Strong Crude Oil","Firm Cash Markets","VeryPositive"]
            }
        },
    ],
    "Soybean Meal": [
        {
            "article": 
            """
            Vegoils commentary: CPO extends gains with strength in related oils and technical buying, soyoil also up
            23 Jul 2025 @ 22:23 UTC
            Vegoils futures traded higher on Wednesday July 23. Crude palm oil (CPO) futures extended gains, supported by strength in related oils and ongoing technical buying. Soyoil futures also moved up, recovering from the previous session’s losses and buoyed by bullish sentiment over US demand.
            The most-active October CPO futures contract on the Bursa Malaysia Derivatives Exchange closed 52 ringgit higher at 4,316 ringgit ($1,022) per tonne, slightly lower than earlier in the day, when the contract rose by 66 ringgit at the end of the morning session.
            Chinese vegoil futures ended Wednesday mixed, with the most-active September palm olein contract on the Dalian Commodity Exchange (DCE) up by 0.74% to 8,994 yuan ($,1256) per tonne, while the equivalent soybean oil futures contract was nearly unchanged after rising by just 2 yuan higher to 8,074 yuan per tonne.
            Rapeseed oil futures on the Zhengzhou Commodity Exchange, on the other hand, closed in negative territory with the most-active contract down by 61 yuan per tonne to 9,456 yuan per tonne, while soyoil futures on the Chicago Mercantile Exchange were trading higher during Asia hours after closing lower overnight.
            The Malaysian ringgit strengthened further against the US dollar on Wednesday, keeping CPO more expensive for buyers trading in dollars.
            CPO prices have traded in a wide range over the past week, fluctuating daily amid a mix of speculative activity, fundamentals and technical factors.
            Meanwhile, July 1-20 production estimates from the Malaysian Palm Oil Association (MPOA) were reported 11.24% higher on the month, a larger increase compared with estimates from brokerage UOB Kay Hian, which were reported 5-9% higher, though the higher output indications had limited effect on CPO futures' buying momentum.
            The Indonesian Palm Oil Association (Gapki) has also reported May palm oil stocks at 2.916 million tonnes, a decline of 130,000 tonnes from April’s 3.046 million tonnes, following a strong export performance with outbound shipments estimated at 2.664 million tonnes, sharply up from 1.779 million tonnes in April.
            Lower production also contributed to the drop in stocks, with output at 4.561 million tonnes, down by 343,000 tonnes from April, while domestic consumption also slipped by 71,000 tonnes to 2.029 million tonnes.
            In the cash market, Chinese buyers were active again with two cargoes traded for August-shipment at $1,060-1,062 per tonne CFR.
            CPO was also reported traded to India at $1,130 per tonne CFR West Coast India (WCI) for September shipment, with discussions for August heard at $1,115-1,130 per tonne CFR WCI and offers for September shipment cargoes at $1,145 per tonne CFR WCI.
            Offers for CPO out of Indonesia were also at $1,090 per tonne FOB Indonesia for August shipment, with a trade heard concluded at $1,085 per tonne FOB for the same month.
            Offers for olein were also heard in the day at $1,032.50 per tonne FOB Indonesia for shipment in the second half of August, with buying ideas around $1,025 per tonne FOB.
            In the Americas, soyoil futures traded higher on the CME, as participants adjusted positions following the previous session’s losses and amid a lingering bullish sentiment over US biofuels demand. Some weakness in crude oil limited part of the positive movement.
            The most-liquid September CME soyoil contract went up by 1.48% on the day to 56.27 cents per lb at 1pm US Eastern time.
            Soymeal futures fell on the CME, mainly pressured downward by product spreading with soyoil and slightly lower corn futures.
            Some strength in soybean prices limited soymeal losses, amid renewed optimism ahead of a meeting between US and China trade officials next week to discuss a potential extension of the deadline for a trade deal.
            Optimism was fueled by recent US trade deals with the Philippines, Indonesia, and most recently, Japan. Under the agreement, Japan is set to invest $550 billion in the US and will see its tariff rate reduced from 25% to 15%.
            “This agreement renews ideas that the US may strike a deal with China," senior agricultural strategist at Marex Terry Reilly said in his daily report.
            The September CME soymeal contract went down by 0.58% on the day to $276.30 per short ton at 1pm US Eastern time.
            In the South American physical market, Fastmarkets heard rumours that Brazilian soyoil for August loading traded at a discount of 4.8 cents per lb to August futures and for September loading at a discount of 5 cents per lb to September futures.
            The basis for September loading was assessed at a discount of 6 cents per lb to September futures.
            In the soymeal front, the September basis in Brazil was assessed at a discount of $9 per short ton to September futures, while in Argentina the corresponding basis was assessed at a discount of $8 per short ton to the same futures contract.
            """,
            "output":{
                "sentiment_score":45,
                "reasoning":"Soymeal prices eased on soyoil-led spreading and weaker corn, but firm soybeans and global demand optimism curbed further losses.",
                "keywords":["Product Spreading","Soybean Oil Strength", "Corn Weakness","Soybean Support", "Meal Price Pressure",]
            }
        },
        {
            "article": 
            """
            News: CBOT soyoil futures end limit-up, again, on US biofuels proposal
            CHICAGO, June 16 (Reuters) - Chicago Board of Trade soybean futures closed steady to higher on Monday as soyoil futures BOv1 rose their daily limit for a second straight session, supported by U.S. biofuel blending proposals announced last week that are likely to increase soyoil demand, traders said.
            Traders continue to digest the U.S. Environmental Protection Agency's larger-than-expected biofuel blending proposals for 2026 and 2027, which were released on Friday and ignited a rally in soyoil. The proposal also included measures to discourage biofuel imports.
            However, benign U.S. crop weather signaled strong harvest prospects that kept a lid on CBOT soybean futures.
            CBOT July soybean futures SN25 settled unchanged at $10.69-3/4 per bushel, easing after recording a one-month high at $10.79-1/4. New-crop November soybeans SX25 settled up 5-3/4 cents at $10.60-1/2.
            CBOT July soyoil BON25 closed up its expanded daily limit of 4.5 cents at 55.11 cents per pound.
            The CBOT said daily limits would remain at expanded levels for Tuesday's trading session.
            CBOT soymeal futures fell on expectations that demand for soyoil would boost the pace of soy crushing, generating surplus meal supplies. July soymeal SMN25 finshed down $8.20 at $283.70 per short ton after hitting a contract low at $289.70.
            The National Oilseed Processors Association said its U.S. members processed 192.829 million bushels of soybeans last month, slightly below trade expectations but the largest May crush ever and the eighth-largest for any month on record.
            NOPA said soyoil stocks among its members as of May 31 fell to 1.373 billion pounds, a bigger drop than most analysts expected.
            Ahead a weekly crop progress report due later on Monday from the U.S. Department of Agriculture, analysts surveyed by Reuters on average expected the government to rate 68% of the U.S. soybean crop in good to excellent condition, unchanged from last week.
            The USDA reported export inspections of U.S. soybeans in the latest week at 215,803 metric tons, in line with trade expectations for 175,000 to 450,000 tons.
            """,
            "output":{
                "sentiment_score":20,
                "reasoning":"Soymeal futures fell to a contract low as higher soyoil demand will spur more crushing, causing surplus meal supply. Strong negative supply shock.",
                "keywords":["Crush Expansion", "Surplus Meal Supply","Biofuel Policy","Negative Spillover"]
            }
        },
        {
            "article": 
            """
            In the Americas, soyoil futures fell on the CME, as sharp losses from the soybean complex weighed on prices, amid new threats of US tariffs on nations dealing with a group of developing countries. Some strength in crude oil limited part of the negative pressure.
            The front-month August CME soyoil contract went down by 1.19% on the day to 53.90 cents per lb at 1pm US Eastern time.
            US President Donald Trump, said the country would impose an additional 10% tariff on nations that aligned with policies of the group of developing nations BRICS ( Brazil, Russia, India, China, South Africa, Egypt, Ethiopia, Indonesia, Iran and the United Arab Emirates) that could potentially go against US interests.
            The US had set Wednesday July 9 as the deadline for many countries to reach agreements on trade deals. But US officials said lately that tariffs would be effective on August 1.  
            Crude oil prices went up in the session, amid a weakness in the US Dollar Index, expectations on interest rates cut by the US Federal Reserve by the end of the year and prospects of reductions of oil production at a faster pace.
            On Friday July 4, after Fastmarkets' closing time, the Trump signed the “One Big, Beautiful Bill” into law after it was approved in the Senate and the House of Representatives last week, with key updates to the 45Z Clean Fuel Production Credit (CFPC). The legislation extends 45Z by two years through the end of 2029 and limits credit eligibility to biofuels produced in the US, Mexico and Canada
            Soymeal futures declined on the CME, pressured downward by a sharp drop in soybean, wheat and corn prices, amid renewed trade tensions. The US favorable weather for crops was also bearish for grain prices in the session.
            The Argentine soymeal cargo sold to China was confirmed on line-up data seen by Fastmarkets. The 30,000-tonne cargo is expected to arrive for loading in a terminal of Up River ports on July 16. Fastmarkets had anticipated the cargo was sold by Bunge, information that was confirmed in the line-up data, at $360 per tonne CFR and split between five buyers, with delivery into Guangdong expected around September.
            """,
            "output":{
                "sentiment_score":30,
                "reasoning":"Soymeal declined with sharp losses in soybeans, wheat, and corn, driven by renewed trade tensions and favorable crop weather (supply↑).",
                "keywords":["Soybean Complex Weakness","Trade Tensions","US Crop Weather","Export Sales","Supply Pressure","VeryNegative"]
            }
        },
        {
            "article": 
            """
            U.S. soymeal futures climbed on Monday, tracking sharp gains in soybeans as persistent drought conditions in Brazil supported the oilseed complex.
            The most-active December soymeal contract at the Chicago Board of Trade settled up $8.30 at $320.10 per short ton, its highest level in three weeks.
            Soybean futures posted their largest daily gain since August, lifted by concerns over reduced output from the world’s top exporter, Brazil.
            “There’s little fundamental news specific to meal,” said one U.S. trader. “But soymeal prices are following soybeans higher, as processors are expected to maintain a firm crush pace.”
            Corn and wheat futures also advanced modestly, with dry weather in the U.S. Midwest providing some underlying support.
            Funds were net buyers of both soybean and soymeal futures, according to trade estimates.
            """,
            "output":{
                "sentiment_score":65,
                "reasoning":"Soymeal prices rose mainly on strength in soybeans and fund buying, with no meal-specific supply/demand news; follows complex higher.",
                "keywords":["Soybean Spillover", "Fund Buying", "Brazil Drought", "Processor Crush Pace"]
            }
        },
    ],
    "Soybean Oil": [
        {
            "article": 
            """
            Wheat surges by daily limit, soyoil hits record high after Russia invades Ukraine
            Reuters News
            25 Feb 2022 03:10:17
            CHICAGO, Feb 24 (Reuters)
            * Black Sea grain supplies uncertain after invasion
            * Wheat highest since 2012, corn at eight-month peak
            * Soybeans slip from 9-1/2 year high on profit taking
            * Soyoil futures hit all-time high on vegoil supply concerns
            (New throughout, updates prices, market activity and comments, adds soyoil futures at record high)
            By Karl Plume
            CHICAGO, Feb 24 (Reuters) - U.S. wheat futures spiked by their daily trading limit on Thursday to their highest since mid-2012 and corn futures rallied to eight-month peaks after Russian forces attacked Ukraine, exacerbating worries over global grain supplies.
            Soyoil futures notched an all-time high on concerns about global vegetable oil supplies amid conflict in the major sunflower oil producing region. Soybean futures eased on profit-taking after setting fresh 9-1/2 year highs overnight.
            Russian forces invaded Ukraine by land, air and sea, confirming the worst fears of the West with the biggest attack by one state on another in Europe since World War Two. [nL1N2UZ089]
            Russia and Ukraine account for about 29% of global wheat exports, 19% of corn supplies and 80% of sunflower oil exports. Traders worry the conflict could trigger a scramble to replace those supplies. [nL1N2UX0MK]
            Ukraine's military suspended commercial shipping at its ports and Moscow suspended the movement of commercial vessels in the Azov Sea until further notice, though it kept Russian ports in the Black Sea open. [nL1N2UZ0L3]
            Still, top wheat importer Egypt cancelled its latest purchasing tender after receiving just one offer after the invasion. [nL1N2UZ1OC]
            "With the ports shut down, that takes a big chunk of grain off the global market and that might send more business to the U.S.," said Ted Seifried, chief agriculture strategist for the Zaner Group.
            Chicago Board of Trade May wheat was up its daily 50-cent trading limit at $9.34-3/4 a bushel by 11:58 a.m. CST (1658 GMT), the highest point for a most-active contract since July 2012.
            May corn was up 14-3/4 cents at $6.96 a bushel after earlier peaking at an eight-month high of $7.16-1/4.
            May soybeans topped at $17.59-1/4 a bushel, the highest for a most-active contract since September 2012, but retreated to $16.66-1/4, down 4-3/4 cents. May soyoil was up 2.45 cents at 73.03 cents per pound after peaking at 74.58 cents, the highest on record for a most-active contract .
            All U.S. wheat, corn, soybean and soyoil contracts posted life-of-contract highs on Thursday.
            (Additional reporting by Sybille de La Hamaide in Paris, Enrico Dela Cruz in Manila and Emily Chow in Beijing Editing by David Goodman, Mark Potter, Diane Craft and David Gregorio) ((karl.plume@thomsonreuters.com; +1 313 484 5285; Reuters Messaging: karl.plume.thomsonreuters.com@reuters.net))
            """,
            "output":{
                "sentiment_score":95,
                "reasoning":"Record high price driven by extreme vegetable oil supply shock. War in major sunflower oil region (80% of exports) forces demand to shift to soyoil.",
                "keywords":["Vegetable Oil Shock","Sunflower Oil Disruption","Record High Price","Demand Shift","Russia-Ukraine War","ExtremePositive"]
            }
        },
        {
            "article": 
            """
            News: CBOT soyoil futures end limit-up, again, on US biofuels proposal
            CHICAGO, June 16 (Reuters) - Chicago Board of Trade soybean futures closed steady to higher on Monday as soyoil futures BOv1 rose their daily limit for a second straight session, supported by U.S. biofuel blending proposals announced last week that are likely to increase soyoil demand, traders said.
            Traders continue to digest the U.S. Environmental Protection Agency's larger-than-expected biofuel blending proposals for 2026 and 2027, which were released on Friday and ignited a rally in soyoil. The proposal also included measures to discourage biofuel imports.
            However, benign U.S. crop weather signaled strong harvest prospects that kept a lid on CBOT soybean futures.
            CBOT July soybean futures SN25 settled unchanged at $10.69-3/4 per bushel, easing after recording a one-month high at $10.79-1/4. New-crop November soybeans SX25 settled up 5-3/4 cents at $10.60-1/2.
            CBOT July soyoil BON25 closed up its expanded daily limit of 4.5 cents at 55.11 cents per pound.
            The CBOT said daily limits would remain at expanded levels for Tuesday's trading session.
            CBOT soymeal futures fell on expectations that demand for soyoil would boost the pace of soy crushing, generating surplus meal supplies. July soymeal SMN25 finshed down $8.20 at $283.70 per short ton after hitting a contract low at $289.70.
            The National Oilseed Processors Association said its U.S. members processed 192.829 million bushels of soybeans last month, slightly below trade expectations but the largest May crush ever and the eighth-largest for any month on record.
            NOPA said soyoil stocks among its members as of May 31 fell to 1.373 billion pounds, a bigger drop than most analysts expected.
            Ahead a weekly crop progress report due later on Monday from the U.S. Department of Agriculture, analysts surveyed by Reuters on average expected the government to rate 68% of the U.S. soybean crop in good to excellent condition, unchanged from last week.
            The USDA reported export inspections of U.S. soybeans in the latest week at 215,803 metric tons, in line with trade expectations for 175,000 to 450,000 tons.
            """,
            "output":{
                "sentiment_score":95,
                "reasoning":"Soyoil futures hit limit-up for 2nd day on larger-than-expected US biofuels proposal, which will sharply boost demand. Surprise, structural impact.",
                "keywords":["US Biofuel Policy","Limit-Up","Demand Surge","EPA Proposal","Import Discouragement", "Structural Bullish"]
            }
        },
        {
            "article": 
            """
            In the Americas, soyoil futures fell on the CME, as sharp losses from the soybean complex weighed on prices, amid new threats of US tariffs on nations dealing with a group of developing countries. Some strength in crude oil limited part of the negative pressure.
            The front-month August CME soyoil contract went down by 1.19% on the day to 53.90 cents per lb at 1pm US Eastern time.
            US President Donald Trump, said the country would impose an additional 10% tariff on nations that aligned with policies of the group of developing nations BRICS ( Brazil, Russia, India, China, South Africa, Egypt, Ethiopia, Indonesia, Iran and the United Arab Emirates) that could potentially go against US interests.
            The US had set Wednesday July 9 as the deadline for many countries to reach agreements on trade deals. But US officials said lately that tariffs would be effective on August 1.  
            Crude oil prices went up in the session, amid a weakness in the US Dollar Index, expectations on interest rates cut by the US Federal Reserve by the end of the year and prospects of reductions of oil production at a faster pace.
            On Friday July 4, after Fastmarkets' closing time, the Trump signed the “One Big, Beautiful Bill” into law after it was approved in the Senate and the House of Representatives last week, with key updates to the 45Z Clean Fuel Production Credit (CFPC). The legislation extends 45Z by two years through the end of 2029 and limits credit eligibility to biofuels produced in the US, Mexico and Canada
            Soymeal futures declined on the CME, pressured downward by a sharp drop in soybean, wheat and corn prices, amid renewed trade tensions. The US favorable weather for crops was also bearish for grain prices in the session.
            The Argentine soymeal cargo sold to China was confirmed on line-up data seen by Fastmarkets. The 30,000-tonne cargo is expected to arrive for loading in a terminal of Up River ports on July 16. Fastmarkets had anticipated the cargo was sold by Bunge, information that was confirmed in the line-up data, at $360 per tonne CFR and split between five buyers, with delivery into Guangdong expected around September.
            """,
            "output":{
                "sentiment_score":35,
                "reasoning":"Soyoil fell as trade tensions and new US tariffs weighed on the soybean complex. 45Z credit limits offer some support, but bearish overall.",
                "keywords":["US Tariffs","Trade Tensions","Soybean Complex Weakness","45Z Policy","Crude Oil Support","MildlyNegative"]
            }
        },
        {
            "article": 
            """
            European vegoils-Palm oil dips on weak soyoil futures, lower energy prices
            LSEG Commodities Research & Forecast
            28 Jan 2020 02:50:12
            ROTTERDAM, Jan 27 (Refinitiv) - Palm oil on the European vegetable oils market dropped further on Monday on the back of a steep fall in soyoil futures and because of a dip in energy prices. Headlines
            The European cash market received no direction from Malaysian palm oil futures, which were closed for Lunar New Year and will reopen on Tuesday.
            Asking prices for palm oil were mostly between $10 and $20 a tonne down from Friday, with a stronger dollar also weighing on dollar priced products.
            At 1730 GMT, CBOT soyoil futures were between 0.62 and 0.73 cents per lb lower on concerns that the corona-virus outbreak in China could hamper U.S. soy exports, while the sharp fall in energy prices, which could cap demand for vegetable oils from biodiesel producers. 0#BO:0#CL:0#LCO:
            EU rapeoil was quoted between eight and 12 euros per tonne down from Friday, following the weak tone in soyoil futures and because of a sharp fall in rapeseed futures, tracking weaker Chicago soybeans. 0#COM:
            Lauric oils were mostly quoted between unchanged and $25 a tonne lower, following the weakness in rival oils and pressure from a stronger dollar.
            "Business was slow all over as the weaker prices discouraged buyers to enter the market," one broker said.
            EUROPEAN VEG OILS TRADES CRUDE PALM OIL, Sumatra/Malaysia sellers option dollars a tonne
            """,
            "output":{
                "sentiment_score":30,
                "reasoning":"Soyoil futures declined on concerns over China coronavirus outbreak, weak export demand, and sharp fall in energy prices.",
                "keywords":["China Demand Risk","Export Concern","Energy Prices","Coronavirus"]
            }
        },
    ],
    "Palm Oil": [
        {
            "article": 
            """
            Indonesia to implement B40 palm oil biodiesel on Jan 1, official says
            Reuters
            22 Aug 2024 20:10:17
            JAKARTA, Aug 22 (Reuters) - Indonesia plans to implement biodiesel with a mandatory 40% blend of palm oil-based fuel from Jan. 1 next year, a senior energy ministry official said on Thursday, lifting prices of the vegetable oil to a more than two-week high.
            Indonesia currently uses B35, which has a 35% blend of palm oil-based biodiesel, and the government had previously said it would increase the blend to 40% next year without specifying a start date.
            "There is no issue in terms of supply volume and other aspects, so we are ready for a mandatory (implementation)," energy ministry senior offical Eniya Listiani Dewi told Reuters.
            Malaysia's benchmark palm oil futures FCPOc3 rose 1.92% to 3,826 ringgit ($874.31) a metric ton on Thursday, its biggest daily gain since July 2.
            "Today's rally is largely driven by Indonesia's announcement of B40. The market knew that it was coming but it is the announcement which has driven prices higher today," said one Singapore-based vegetable oils analyst
            When implemented, the B40 mandate could increase biodiesel consumption to up to 16 million kilolitres (KL) next year from a forecast of 13 million KL this year, the energy ministry has estimated.
            B40 will boost Indonesia's palm oil use for biodiesel to 13.9 million metric tons, from the estimated 11 million tons needed this year with B35, Indonesia's biofuel producers association APROBI had previously estimated.
            Domestic palm oil consumption has grown 7.6% on average since 2019, driven in part by policies including biodiesel mandates and compulsory domestic sales for cooking oil, while output has risen less than 1% annually according to GAPKI data.
            Indonesia's biggest palm oil producers association GAPKI had warned that a higher mix could hurt exports amid stagnating production.
            The regulation stating the B40 mandate is yet to be issued.
            """,
            "output":{
                "sentiment_score":85,
                "reasoning":"Strongly bullish as top producer Indonesia officially announced a B40 biodiesel mandate starting Jan 1. This creates significant, structural new demand, tightening the global market.",
                "keywords":["Indonesia Biofuel Mandate", "B40", "Structural Demand", "Export Reduction", "VeryPositive"]
            }
        },
        {
            "article": 
            """
            VEGOILS-Palm subdued on higher output, stronger ringgit
            Reuters News
            24 Jul 2025 14:26:23
            Updates for mid-session trading, adds trader's comment
            KUALA LUMPUR, July 24 (Reuters) - Malaysian palm oil futures were subdued on Thursday as higher production and a stronger ringgit dampened sentiment despite gains in Chicago soyoil.
            The benchmark palm oil contract FCPOc3 for October delivery on the Bursa Malaysia Derivatives Exchange was down 6 ringgit, or 0.14%, at 4,309 ringgit ($1,022.30) a metric ton by the midday break. The contract rose in the last two sessions.
            "The higher-than-expected production scenario has halted the rally in palm oil prices, while the ringgit continues to strengthen against the U.S. dollar, which is also contributing to the decline in the ringgit-denominated contract," said Anilkumar Bagani, research head at Sunvin Group.
            Adding to the cautious mood, fresh palm oil purchases by India have slowed down due to a sharp surge in prices, Bagani said.
            Dalian's most-active soyoil contract DBYcv1 rose 0.55%, while its palm oil contract DCPcv1 added 0.27%. Soyoil prices on the Chicago Board of Trade BOcv1 were up 0.02%.
            Palm oil tracks price movements of rival edible oils, as it competes for a share of the global vegetable oils market.
            The ringgit MYR=, palm's currency of trade, strengthened 0.24% against the dollar, making the commodity more expensive for buyers holding foreign currencies.
            Oil prices rose, buoyed by optimism over U.S. trade negotiations that would ease pressure on the global economy and a sharper-than-expected decline in U.S. crude inventories. O/R
            Stronger crude oil futures make palm a more attractive option for biodiesel feedstock.
            Indonesia's palm oil stocks at the end of May had contracted by 4.27% from the previous month to 2.9 million metric tons after a surge in exports, data from the Indonesia Palm Oil Association showed.
            """,
            "output":{
                "sentiment_score":40,
                "reasoning":"Price slightly down as higher-than-expected production, a stronger ringgit, and slowing Indian demand (Supply↑, Demand↓) offset support from stronger rival oils and crude.",
                "keywords":["Higher Production","Ringgit Strength","Indian Demand","Rival Edible Oils","Mixed Signals","MildlyNegative"]
            }
        },
        {
            "article": 
            """
            VEGOILS-Palm oil falls 1.5% as COVID restrictions curb Chinese demand
            Reuters News
            08 Nov 2022 19:35:41
            SINGAPORE, Nov 8 (Reuters) - Malaysian palm oil futures slid more than 1% on Tuesday, falling for the first time in three sessions, on concerns over tepid demand in key consumer China amid COVID-19 restrictions.
            The benchmark palm oil contract on the Bursa Malaysia Derivatives Exchange closed down 65 ringgit, or 1.5%, at 4,368 ringgit ($922.69) a tonne.
            "There were hopes that China would move from zero-COVID policy," said one analyst. "But authorities have ruled out a shift from that policy."
            The global outlook for palm oil remains uncertain, with strict pandemic policies in major importer China weighing on demand, while high energy prices and a slowdown in output provide support, leading industry analysts said at a conference on Friday.
            China will persevere with its "dynamic-clearing" approach to COVID-19 cases as soon as they emerge, health officials said on Saturday, adding that measures must be implemented more precisely and meet the needs of vulnerable people.
            China reported 7,691 new COVID-19 infections on Nov. 7, of which 890 were symptomatic and 6,801 were asymptomatic, the National Health Commission said on Tuesday.
            Malaysia's benchmark palm oil contract is expected to trade between 3,500 ringgit and 4,500 ringgit per tonne until the end of next March, leading industry analyst Dorab Mistry said.
            Palm oil may test a support at 4,311 ringgit a tonne, a break below could open the way towards 4,220 ringgit-4,264 ringgit range, according to Wang Tao, a Reuters analyst for commodities technicals.
            In related edible oils, Dalian's most-active soyoil contract lost 1.1%, while its palm oil contract added 0.3%. ($1 = 4.7340 ringgit)
            """,
            "output":{
                "sentiment_score":40,
                "reasoning":"Price fell as hopes for China easing its zero-COVID policy were dashed. Persistent restrictions in a key importing nation are curbing demand, outweighing general supportive factors.",
                "keywords":["China Demand","Zero-COVID Policy","Import Concerns","MildlyNegative","DemandCurb"]
            }
        },
        {
            "article": 
            """
            VEGOILS-Palm hits 9-week closing high on tight supply view, flood woes
            Reuters News
            05 Jan 2022 19:25:06
            KUALA LUMPUR, Jan 5 (Reuters)
            * Palm oil hits highest closing since Nov. 3
            * Dec stocks pegged to decline as output slumps
            * Refinitiv cuts output forecast as floods hit Indonesia, Malaysia
            (Updates with closing prices, adds details)
            By Mei Mei Chu
            KUALA LUMPUR, Jan 5 (Reuters) - Malaysian palm oil futures ended Wednesday at a nine-week high, underpinned by forecasts of a drop in December inventory levels and worries over floods disrupting output in the world's second-largest producer of the commodity.
            The benchmark palm oil contract for March delivery on the Bursa Malaysia Derivatives Exchange closed up 123 ringgit, or 2.5%, to 5,037 ringgit ($1,201.57) a tonne.
            It rose for a fourth consecutive session to its highest closing since Nov. 3.
            Malaysia's palm oil inventories at end-December likely shrunk 4.9% from the previous month to 1.73 million tonnes, their lowest in five months, a Reuters survey ahead of Malaysian Palm Oil Board data showed. [PALM/POLL]
            Production is pegged to fall 8.6% to 1.49 million tonnes as floods hamper output, while exports are seen declining 4.9% to 1.4 million tonnes.
            The disruption in arrivals of palm fruit bunches at flood-prone areas has resulted in millers delaying deliveries to refineries, said Paramalingam Supramaniam, director at Selangor-based brokerage Pelindung Bestari.
            Refinitiv Commodities Research in a note lowered its 2021/22 output forecasts for top producers Indonesia and Malaysia by less than 1% from its last update due to flooding risks in the two countries. [nPGEvDYYca]
            "Looking ahead, the developments of labour shortages in Malaysia, Delta/Omicron variants of COVID-19, soybean developments in South America and, persistent weather concerns led by La Niña across Southeast Asia to South America are some of the key swing drivers," Refinitiv said.
            Malaysia plans to implement its nationwide adoption of the B20 palm oil biofuel programme by the end of 2022, the country's palm oil board said. [nK7N2OZ00P]
            Dalian's most-active soyoil contract rose 1.2%, while its palm oil contract gained 1.8%. Soyoil prices on the Chicago Board of Trade were up 1.3%.
            Palm oil is affected by price movements in related oils as they compete for a share in the global vegetable oils market.
            ($1 = 4.1920 ringgit)
            """,
            "output":{
                "sentiment_score":80,
                "reasoning":"Strong rally to 9-week high driven by a clear supply-side narrative. Floods in Malaysia & Indonesia are disrupting production and lowering inventory forecasts, outweighing export concerns.",
                "keywords":["Flood Disruption", "Supply Tightness", "Inventory Decline", "Production Forecast Cut", "La Niña", "VeryPositive"]
            }
        },
    ]
}

MASTER_PROMPT_TEMPLATE = """
### Your Role & Objective ###
You are a highly specialized {commodity_name} market analysis AI. Analyze the provided news article and market context to produce a sentiment score (0-100), a concise reasoning (English, 150 chars max), and relevant keywords in a strict JSON format.

### ANALYSIS GUIDELINES ###

- **Principle 1: Good for the Crop is BAD for the Price.** Favorable weather, high crop condition ratings (e.g., "good to excellent"), and smooth harvesting increase supply (Supply↑). This is fundamentally BEARISH. The score must be below 50.
- **Principle 2: Supply Shocks are GOOD for the Price.** Port attacks, export bans, droughts, floods, or poor crop conditions reduce supply (Supply↓). This is fundamentally BULLISH. The score must be above 50.
- **Principle 3: Geographical Weighting:** The impact on global benchmark prices (like CBOT) is not equal across all countries. Prioritize news based on the commodity-specific tiers below. State the relevant region/country in your reasoning.
    **# For {commodity_name} = "Corn"**
    - **Tier 1 (Highest Impact):** USA
    - **Tier 2 (High Impact):** Brazil, Argentina
    - **Tier 3 (Medium Impact):** Ukraine, China

    **# For {commodity_name} = "Wheat"**
    - **Tier 1 (Highest Impact):** USA, Russia
    - **Tier 2 (High Impact):** Ukraine, Black Sea Region, EU (especially France, Germany)
    - **Tier 3 (Medium Impact):** Canada, Australia, Argentina, India, China

    **# For {commodity_name} = "Soybeans"**
    - **Tier 1 (Highest Impact):** USA, Brazil
    - **Tier 2 (High Impact):** Argentina
    - **Tier 3 (Medium Impact):** China (as importer)

    **# For {commodity_name} = "Soybean Meal"**
    - **Tier 1 (Highest Impact):** Argentina
    - **Tier 2 (High Impact):** Brazil, USA
    - **Tier 3 (Medium Impact):** EU, Southeast Asia (as importers)

    **# For {commodity_name} = "Soybean Oil"**
    - **Tier 1 (Highest Impact):** USA (due to its role as the primary price driver via biofuel policies)
    - **Tier 2 (High Impact):** Argentina, Brazil (due to physical export volume)
    - **Tier 3 (Medium Impact):** India, China (as importers)

    **# For {commodity_name} = "Palm Oil"**
    - **Tier 1 (Highest Impact):** Indonesia, Malaysia
    - **Tier 2 (High Impact):** India, China (as importers)
    - **Tier 3 (Medium Impact):** Thailand, Colombia
- **Principle 4: News Fatigue:** For ongoing, long-term events like a war, a new development (e.g., a specific port attack) is still significant but has less impact than the initial outbreak. Do not assign extreme scores (like 90+) unless the article frames it as a major escalation.

**II. Market Sentiment Scoring Scale (0-100):**
- 0-10 (Extreme Negative): Catastrophic events, systemic shocks mentioned in the text.
- 11-30 (Very Negative): Strong, fundamental downward pressure.
- 31-49 (Mildly Negative): Slightly bearish news, headwinds.
- 50 (Neutral): In line with expectations, already priced in.
- 51-69 (Mildly Positive): Slightly bullish news, tailwinds.
- 70-89 (Very Positive): Strong, fundamental upward pressure.
- 90-100 (Extreme Positive): Paradigm-shifting positive events mentioned in the text.

**III. Core Evaluation Factors:**
The final score is determined by synthesizing these factors based only on the provided text. A and B are the primary drivers.

A. Supply/Demand Impact: Based on the text, state the anticipated shifts (e.g., Supply↓, Demand↑). Prioritize concrete data or facts mentioned in the article.

B. Expectedness vs. Surprise: Does the article state or imply that the event was unexpected? (e.g., "surpassing expectations", "a surprise announcement", "contrary to analyst predictions"). High surprise, if mentioned, amplifies the score. If not mentioned, assume it is neutral.

C. Magnitude, Scope, & Duration: Assess the impact's scale as described within the text.
- Magnitude Scale (Qualitative): Judge based on the adjectives used in the article.
- Extreme: "historic", "unprecedented", "catastrophic", "record-breaking", "drastic"
- Strong: "major", "significant", "sharp", "substantial"
- Moderate: "modest", "slight", "better/worse than feared"
- Scope: Is the event described as regional or global?
- Duration: Is the event described as a short-term disruption or a long-term structural change?

D. Information Quality: Is the source cited in the article an official body (USDA, EPA), a private firm, or an unnamed source/rumor? Give more weight to news citing official sources.

**IV. Advanced Interpretation & Tie-breaking:**
Explicit Contextual Mentions: If the news text explicitly mentions macro factors (e.g., "a strengthening dollar is pressuring exports") or competing commodities (e.g., "demand is shifting from corn to wheat due to high prices"), reflect this directly in your supply/demand analysis. If not mentioned, do not infer them.
Factor Weighting: In most cases, the score should be driven by the Supply/Demand Impact (A) and the level of Surprise (B). Other factors (C, D) help to fine-tune the score within the range determined by A and B.

**V. Output Generation Guidelines:**
Reasoning Field: Clearly explain how you arrived at the score by referencing the core factors above strictly under 150 characters.
Keywords Field: Keywords MUST be specific factors impacting the commodity price (e.g., 'ethanol demand', 'export policy', 'drought conditions'). They must NOT be the commodity name itself (e.g., 'Corn') and must EXCLUDE other commodities mentioned only in passing.


### ANALYSIS TASK ###

**News Article for Your Analysis:**
{news_article_text}

Output your analysis as a single, valid JSON object ONLY, based on all guidelines and examples.

"""

# |--------------------------------|
# |--- 3. DB 관련 헬퍼 함수 정의 ---|
# |--------------------------------|

def fetch_commodities(cur):
    """
    commodities 마스터 테이블에서 분석 대상이 되는 모든 품목 목록을 가져옵니다.
    이 목록은 GPT 프롬프트를 동적으로 생성하는 데 사용됩니다.
    Args:
        cur (psycopg2.cursor): 데이터베이스 커서 객체
    Returns:
        list: 품목 이름 문자열의 리스트 (예: ['Corn', 'Wheat', ...])
    """
    cur.execute("SELECT name FROM commodities ORDER BY name;")
    return [row[0] for row in cur.fetchall()]

def fetch_news_to_analyze(cur, limit=5):
    """
    raw_news 테이블에서 아직 분석되지 않은 (analysis_status = FALSE) 뉴스들을
    지정된 개수(limit)만큼 가져옵니다.
    Args:
        cur (psycopg2.cursor): 데이터베이스 커서 객체
        limit (int): 한 번에 가져올 뉴스의 최대 개수
    Returns:
        list: (id, title, content) 튜플의 리스트
    """
    #DB 스키마대로! 
    cur.execute("SELECT id, title, content FROM raw_news WHERE analysis_status = FALSE LIMIT %s;", (limit,))
    return cur.fetchall()

def create_few_shot_prompt(commodity_name):
    """
    [최종 수정안]
    Few-shot 예시의 JSON 문자열에서 중괄호를 이중으로(escape)
    LangChain 파서가 변수로 오인하는 것을 원천적으로 차단.
    """
    messages = [
        ("system", f"You are a highly specialized {commodity_name} market analysis AI. Your primary function is to interpret news data with nuance and objectivity, adhering strictly to the provided guidelines and examples.")
    ]

    examples = FEW_SHOT_EXAMPLES.get(commodity_name, [])
    for example in examples:
        messages.append(("user", example['article']))
        
        # ★★★ 이중 중괄호로 JSON 문자열을 감싸서 파싱 오류를 회피. ★★★
        escaped_output = json.dumps(example['output']).replace('{', '{{').replace('}', '}}')
        messages.append(("assistant", escaped_output))

    messages.append(("user", MASTER_PROMPT_TEMPLATE))

    return ChatPromptTemplate.from_messages(messages)


# --- 4. 메인 분석 및 저장 로직 ---

def analyze_and_store_all_news():
    """
    전체 분석 파이프라인을 실행하는 메인 함수
    1. 분석 대상 뉴스 조회
    2. 뉴스별 관련성 필터링
    3. 관련 품목 분류
    4. 품목별 감성 분석
    5. DB에 최종 결과 저장
    """
    conn = db_pool.getconn()
    try:
        with conn.cursor() as cur:
            # --- 분석에 필요한 사전 정보 준비 ---
            commodities_list = fetch_commodities(cur)
            candidate_list_str = "\n- ".join(commodities_list)
            news_items = fetch_news_to_analyze(cur)
            print(f"Found {len(news_items)} news articles to analyze.")

            # --- LangChain 체인(Chain) 정의 ---

            # [체인 1: 선물 시장 관련성 필터링]
            # 뉴스가 분석할 가치가 있는지 가장 먼저 판단.
            relevance_prompt = ChatPromptTemplate.from_template(
                """Is the following news article directly relevant to the global market price or supply-demand fundamentals of agricultural/energy/metal futures (CBOT, CME, ICE, LME, Euronext, etc.)?

                Say "YES" if the article covers:
                - Any change in global or regional supply, demand, production, consumption, stocks, weather (for agriculture/softs), mining/output/processing (for metals/energy), shipping/logistics, or exports/imports that could reasonably affect futures or spot prices
                - Government, regulatory, or international policies/actions (e.g. tariffs, quotas, sanctions, taxes, subsidies, trade agreements, environmental rules, central bank decisions) impacting supply, demand, or price
                - Major geopolitical events, wars, strikes, port/plant/mine shutdowns, natural disasters, epidemics, or similar shocks
                - Release of important market data, official reports, or statistics (e.g. USDA WASDE, EIA, CFTC, Crop Progress, OPEC meetings, PMI, CPI, GDP, etc.)
                - Announcements from **major producers, exporters, importers, or traders** (companies, SOEs, or countries) that could shift market supply/demand balance (e.g. large production cuts/increases, export bans, capacity expansions/closures, force majeures), **only if such entities have enough market power to move prices**
                - Significant financial flows, speculative activity, large fund positioning, futures/options/basis/structure changes, or market rumors influencing price
                - Major technical analysis, price chart patterns, signals, or trend changes (such as moving average crossovers, head-and-shoulders, support/resistance, wave analysis, RSI, MACD, or analyst technical calls) that could influence market trading, sentiment, or trigger significant moves
                - Any other news likely to affect the trading price, volatility, or global trade/sentiment for its close substitutes/related markets

                Say "NO" if the article is about:
                - Processed foods, finished products, recipes, consumer brands, retail trends, marketing, health/nutrition, restaurants, cafes, electronics, jewelry, fashion, or product launches
                - Company news/results/sales/marketing that do **not** involve major changes to global/regional supply, demand, or price (e.g. earnings, new snack products, retail expansion, new store openings)
                - General lifestyle, culture, sports, entertainment, science, technology trends, medical news, or other topics **not likely to affect market price or supply-demand**

                Title: {news_title}
                Body: {news_content}

                Answer: YES or NO only."""
            )
            relevance_chain = relevance_prompt | llm | StrOutputParser()

            # [체인 2: 품목 분류]
            # 관련성이 확인된 뉴스에서, 어떤 주제의 내용인지 찾아냄

            classification_prompt = ChatPromptTemplate.from_template(
            """
            Your task is to identify the main subject commodities of this article, based on the depth of discussion.

            A commodity is a **"main subject"** and should be INCLUDED if the article provides specific, detailed analysis about its own market. This includes:
            - Its own specific price changes (e.g., 'Corn futures settled down 1/2-cent').
            - Dedicated paragraphs discussing its unique supply or demand factors (e.g., yield reports for wheat, weather for corn, specific import/export news for soybeans).
            - A specific outlook or analysis for that commodity.

            A commodity is a **"minor factor"** and should NOT be included if it is only mentioned in passing to explain another commodity's market (e.g., 'stronger soyoil prices supported palm oil').

            ### Additional Notes & Special Rules ###
            You must apply these rules to override the general guidelines above when applicable.

            **1. Product Chain & Alias Rule:**
            - **Soybean -> Soybean Meal :** If the main subject is "Soybean", you **MUST ALWAYS** include "Soybean Meal". Their markets are tightly linked because meal is a direct co-product of the soybean crushing process, so any news about the source bean is critical for the meal market.
            - **Soybean -> Soybean Oil :** If the main subject is "Soybean", include "Soybean Oil" **ONLY IF** the article also discusses themes relevant to vegetable oils, such as **biodiesel, renewable fuels, competing oils (e.g., palm oil), or crude oil price impacts.** Otherwise, do not automatically include "Soybean Oil".
            - **Aliases & Variants :** Treat common names, abbreviations, or specific variants as their official candidate name.
            - **Example 1:** If the article mentions "soymeal", you should classify "Soybean Meal".
            - **Example 2:** Articles about "soft wheat", "hard red winter wheat", or "spring wheat" must all be classified as "Wheat".

            **2. Geopolitical & Macro Rule (with INFERRED impact):**
            - When an article discusses a major event (war, natural disaster) or a critical macro issue (tariffs, sanctions) without explicitly naming commodities, you **must use your knowledge as a market analyst to infer the most impacted commodities.**
            - **Your reasoning process should be:**
                1.  Identify the countries or regions involved (e.g., U.S. and China, Brazil, Black Sea Region).
                2.  Based on your knowledge, determine which commodities on the candidate list are the most significant exports/imports for that region or are most exposed to that type of event.
                3.  Classify those inferred commodities as "main subjects."
            - **Example:** For a broad "U.S.-China tariff" article, you should reason that major U.S. agricultural exports to China are at risk, and therefore classify "Soybean", "Corn", and "Wheat". For a major "drought in Argentina" article, you should classify "Soybean" and "Corn".

            **3. Technical Analysis Rule :**
            - If an article's main focus is the **technical analysis** of a commodity (e.g., mentioning "technical selling," chart patterns, support/resistance levels), you MUST classify that commodity.

            Based on all rules, analyze the article and return a JSON array of all main subject commodities from the candidate list. If multiple commodities are discussed as main subjects, include all of them.

            **Candidate Commodities:**
            {candidate_list}

            **Example 1 (Single Main Subject):**
            - Article: Focuses on Palm Oil's market, but mentions that rising Soyoil prices are providing support.
            - Correct Output: ["Palm Oil"]

            **Example 2 (Multiple Main Subjects):**
            - Article: Discusses North Dakota's yield reports for Wheat in one section, and crop-friendly weather for Corn in another section.
            - Correct Output: ["Wheat", "Corn"]

            **Article for Your Analysis:**
            Title: {news_title}
            Content: {news_content}
            """
)
            classification_chain = classification_prompt | llm | JsonOutputParser()

            # --- 분석할 뉴스가 없을 때까지 5개씩 반복 처리 ---
            while True:
                news_items = fetch_news_to_analyze(cur, limit=5)
                if not news_items:
                    logging.info("분석할 새로운 뉴스가 없습니다. 작업을 종료.")
                    break

                logging.info(f"총 {len(news_items)}개의 뉴스를 배치 처리.")

                # --- 뉴스 아이템별 분석 루프 시작 ---
                for news_id, title, content in news_items:
                    logging.info(f"--- News ID: {news_id} 분석 시작 ---")
                    news_text = f"Title: {title}\n\nBody: {content}"

                    try:
                        # 1. 선물 시장 관련성 필터링 실행
                        relevance_decision = relevance_chain.invoke({
                            "news_title": title,
                            "news_content": content
                        })

                        if "NO" in relevance_decision.upper():
                            logging.info(f"   > News ID {news_id}: 관련 없는 뉴스로 판단되어 건너뜁니다.")
                            cur.execute("UPDATE raw_news SET analysis_status = TRUE, relevant_news = FALSE WHERE id = %s;", (news_id,))
                            conn.commit()
                            continue

                        # 2. 관련성 있는 뉴스일 경우, 품목 분류 실행
                        classified_commodities = classification_chain.invoke({
                            "candidate_list": candidate_list_str,
                            "news_title": title,
                            "news_content": content
                        })

                        if not classified_commodities:
                            logging.info(f"   > News ID {news_id}: 관련은 있으나, 지정된 품목이 없어 건너뜁니다.")
                            cur.execute("UPDATE raw_news SET analysis_status = TRUE, relevant_news = TRUE WHERE id = %s;", (news_id,))
                        else:
                            logging.info(f"   > News ID {news_id}: 관련 품목 {classified_commodities} 발견.")
                            for commodity_name in classified_commodities:
                                if commodity_name not in commodities_list:
                                    logging.warning(f"     - '{commodity_name}'은 마스터 목록에 없는 품목입니다. 건너뜁니다.")
                                    continue

                                logging.info(f"     - '{commodity_name}'에 대한 감성 분석 실행...")
                                
                                # 3-1. commodity_name에 맞는 프롬프트 '템플릿' 생성
                                sentiment_prompt = create_few_shot_prompt(commodity_name)
                                
                                # 3-2. LangChain 체인 구성
                                sentiment_chain = sentiment_prompt | llm | JsonOutputParser()
                                
                                # 3-3. invoke 시점에 플레이스홀더에 들어갈 변수들을 딕셔너리로 전달
                                sentiment_result = sentiment_chain.invoke({
                                    "commodity_name": commodity_name,
                                    "news_article_text": news_text
                                })
                                
                                # 4. 분석 결과 DB에 저장 (이 부분은 기존 코드와 동일)
                                cur.execute("SELECT id FROM commodities WHERE name = %s;", (commodity_name,))
                                commodity_id = cur.fetchone()[0]

                                cur.execute("INSERT INTO news_commodity_link (raw_news_id, commodity_id) VALUES (%s, %s) ON CONFLICT DO NOTHING;", (news_id, commodity_id))
                                
                                cur.execute(
                                    "INSERT INTO news_analysis_results (raw_news_id, commodity_id, sentiment_score, reasoning, keywords) VALUES (%s, %s, %s, %s, %s);",
                                    (
                                        news_id,
                                        commodity_id,
                                        sentiment_result.get('sentiment_score'),
                                        sentiment_result.get('reasoning'),
                                        json.dumps(sentiment_result.get('keywords')),
                                    )
                                )
                                logging.info(f"       > '{commodity_name}' 분석 결과 저장 완료.")

                            # 이 뉴스의 모든 관련 품목 분석이 끝나면, 최종 상태를 업데이트.
                            cur.execute("UPDATE raw_news SET analysis_status = TRUE, relevant_news = TRUE WHERE id = %s;", (news_id,))
                        
                        # 현재 뉴스에 대한 모든 DB 작업을 최종 확정.
                        conn.commit()

                    except Exception as e:
                        logging.error(f"News ID {news_id} 처리 중 에러 발생: {e}", exc_info=True)
                        conn.rollback() # 해당 뉴스에 대한 작업만 롤백

    except (Exception, psycopg2.Error) as error:
        logging.error(f"데이터베이스 작업 중 심각한 에러 발생: {error}", exc_info=True)
    finally:
        if conn:
            # 사용이 끝난 커넥션을 풀에 반납.
            db_pool.putconn(conn)
            logging.info("데이터베이스 커넥션을 풀에 반납했습니다.")

if __name__ == '__main__':
    analyze_and_store_all_news()
    # 스크립트 종료 시 모든 유휴 커넥션을 닫습니다.
    if 'db_pool' in locals() and db_pool:
        db_pool.closeall()
        logging.info("모든 데이터베이스 커넥션이 종료되었습니다.")