"""
rag_faiss.py
------------
Builds a FAISS vector store from a car-pricing knowledge base
and exposes retrieve_docs(query) for semantic search.
"""

import re

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── 1. Knowledge base – domain text about car pricing ─────────────────────────
KNOWLEDGE_BASE = """
Car Pricing Fundamentals
========================
Car prices in India are influenced by multiple interrelated factors. Understanding
these factors helps buyers make informed decisions and sellers set competitive prices.

Brand & Market Positioning
--------------------------
Premium brands like BMW, Mercedes-Benz, and Audi command significantly higher prices
due to brand equity, imported components, and luxury positioning. Mass-market brands
such as Maruti Suzuki, Hyundai, and Honda offer competitive pricing through local
manufacturing and high volumes. Toyota occupies a middle ground, known for reliability
and strong resale value. Brand perception and after-sales service network also affect
pricing.

Brand Snapshots Relevant to the Dataset
---------------------------------------
Honda is commonly associated with practical, refined, and reliable cars in India.
Honda models are often appreciated for smooth petrol engines, comfortable city driving,
good cabin space, and balanced resale value. In used-car discussions, Honda City is
frequently seen as a strong sedan choice, while Amaze is viewed as a value-oriented
compact sedan. Buyers often compare Honda with Hyundai, Maruti Suzuki, and Toyota when
looking for dependable daily-use cars. Honda is often seen as a sensible choice for
buyers who want refinement, comfort, and trust in the sedan segment.

BMW represents the premium and luxury end of the market. BMW cars usually command
higher prices because of premium branding, stronger performance, imported components,
advanced features, and higher maintenance expectations. In resale markets, BMW pricing
depends heavily on model year, kilometres driven, service history, and overall condition.
BMW is often associated with executive sedans, luxury SUVs, sporty driving feel, and
high ownership costs compared with mass-market brands.

Toyota is known for long-term reliability, durability, and strong resale value. Toyota
cars often retain value better than many mainstream brands because buyers trust their
maintenance record and longevity. Toyota is commonly preferred by buyers who prioritise
peace of mind, durability, and strong long-term ownership value.

Hyundai is associated with feature-rich cars, wide market reach, and competitive
pricing. Hyundai models are popular in hatchback, sedan, and SUV segments and are often
considered a good balance of styling, features, and value. Hyundai is often chosen by
buyers who want modern features, attractive design, and good value across segments.

Maruti Suzuki is known for affordability, fuel efficiency, low running costs, and a
very broad service network. In the used-car market, Maruti cars are often easier to
sell because of buyer familiarity and lower maintenance costs. It is especially strong
in hatchbacks and compact cars, and many buyers see it as the most practical brand for
budget-friendly ownership in India.

Ford is often discussed in the used-car market in terms of build quality, driving feel,
and diesel performance. Resale value can vary because buyer demand depends more on model
support perception and local service access.

Tata Motors is strongly associated with value-oriented cars, improving safety perception,
and growing electric vehicle presence in India. Tata models are often discussed in
segments such as hatchbacks, compact SUVs, and EVs. Buyers commonly look at Tata for
strong feature value, road presence, and lower running-cost potential in EV options.

Mahindra is known for SUVs, diesel strength, rugged appeal, and strong road presence.
In the Indian market, Mahindra is often associated with utility-focused vehicles,
off-road capable SUVs, and family-oriented larger vehicles. Resale value is often
supported by strong demand for popular SUV models.

Kia is usually viewed as a modern, feature-rich brand with strong appeal in MPV and SUV
segments. Kia cars are often chosen by buyers who want premium-looking interiors,
technology features, and stylish design at competitive prices.

Renault is often linked with value-focused hatchbacks and compact SUVs. Buyers may see
Renault as practical for budget-conscious ownership, especially when comparing entry-
level and compact-segment cars.

Nissan is generally discussed in terms of selected practical models and value-oriented
ownership. Buyer interest often depends on model reputation, service access, and local
market demand.

Skoda is known for solid build quality, mature styling, and driving comfort. In India,
Skoda is often associated with buyers who want a more premium mainstream experience,
especially in sedan and SUV categories.

Volkswagen is often appreciated for build quality, stability, and driving feel. Buyers
who value European-style dynamics and a more premium feel in mainstream segments often
compare Volkswagen with Skoda and Hyundai.

Audi represents the premium luxury segment, similar to BMW and Mercedes-Benz. Audi cars
are usually associated with high-end features, premium interiors, imported components,
and expensive maintenance expectations. Their resale depends heavily on condition,
service history, and model desirability.

Mercedes-Benz is associated with luxury, comfort, prestige, and premium ownership in
India. Mercedes cars usually command high prices due to brand strength, imported
components, advanced features, and luxury positioning. Resale is influenced by age,
running condition, kilometres driven, and service record.

MG is known in India for feature-loaded SUVs and electric vehicles with strong perceived
value. Buyers often associate MG with technology features, spacious cabins, and a modern
ownership experience in selected segments.

How to Interpret Brand Questions
--------------------------------
If a user asks about a brand such as Honda, BMW, Toyota, Hyundai, Maruti, Ford, Tata,
Mahindra, Kia, Renault, Nissan, Skoda, Volkswagen, Audi, Mercedes-Benz, or MG, the
chatbot should explain the brand's market positioning, typical strengths, and how it
usually affects pricing or resale in India. The answer should stay grounded in the
pricing and used-car domain rather than drifting into unrelated company details.

Year of Manufacture & Depreciation
------------------------------------
A car depreciates fastest in its first year (roughly 15–20%). From year 2–5 the
depreciation stabilises at 8–12% per year. Cars older than 10 years depreciate more
slowly in absolute terms but may face higher maintenance costs. Newer models (2020 and
beyond) typically have better safety ratings and fuel efficiency, adding to their value.

Kilometres Driven (Odometer Reading)
--------------------------------------
Lower kilometres driven generally means less wear and tear. A car with under 30,000 km
is considered low-usage. Between 30,000–80,000 km is average. Above 1,00,000 km is
considered high usage and significantly reduces the price. Commercial-use vehicles
depreciate faster due to higher kilometres.

Fuel Type
---------
Petrol cars are popular for city driving due to lower upfront cost and smoother drive.
Diesel cars cost more upfront but deliver better mileage, making them suitable for
highway and long-distance use. Electric vehicles (EVs) are gaining traction with lower
running costs but higher initial prices. CNG cars are economical but have limited
infrastructure outside major cities.

Engine Displacement & Horsepower
----------------------------------
Engine size (measured in cc) and output (horsepower) directly impact performance and
price. Small engines (800–1200 cc) are economical; mid-range (1200–2000 cc) balance
performance and efficiency; large engines (>2000 cc) are found in SUVs and performance
cars and command premium pricing. Higher horsepower models cost more to insure and
maintain.

Seating Capacity
-----------------
5-seater sedans and hatchbacks are the most common. 7-seater SUVs and MPVs carry a
premium due to versatility and family appeal. Commercial variants (9–12 seaters) are
priced differently and subject to different regulations.

Condition & Service History
----------------------------
A well-maintained car with full service history from authorised service centres fetches
a higher price. Single-owner vehicles are preferred over multi-owner ones. Accident
history, paint condition, tyre condition, and interior quality all affect resale value.

Market Trends in India
-----------------------
As of 2024, SUVs dominate new and used car sales in India. Popular segments include
compact SUVs (Creta, Seltos), mid-size SUVs (XUV700, Thar), and luxury SUVs (Fortuner,
GLC). Electric vehicles from Tata (Nexon EV, Punch EV) and MG (ZS EV) are growing in
popularity. Used-car platforms like CarDekho, Cars24, and OLX Autos have increased
price transparency. Insurance, road tax, and registration charges add 10–15% to the
on-road price over the ex-showroom price.

Car Pricing FAQ
---------------
What factors affect car resale value in India?
Car resale value in India depends on brand reputation, year of manufacture,
kilometres driven, fuel type, service history, accident history, number of previous
owners, tyre condition, interior condition, and current market demand. Cars from
trusted brands with lower usage and clean records usually hold value better.

Why do diesel cars cost more than petrol?
Diesel cars often have a higher upfront price because diesel engines are built to
handle higher compression and long-distance running. They are attractive to heavy
users because of better mileage on highways. Their long-term value also depends on
maintenance cost, usage pattern, and regulation concerns in some regions.

How does mileage (km driven) affect car price?
Mileage is one of the strongest signals of wear and tear. Lower kilometres driven
usually support a higher price because the car is seen as less used. Average usage
gets balanced pricing, while very high mileage reduces resale value because buyers
expect more maintenance and lower remaining life.

Tips for buying a used car in India
Inspect the car carefully, verify the RC and ownership details, check service
history, confirm insurance status, look for accident repairs, compare prices across
multiple platforms, and take a proper test drive. It is also important to check
whether any loan or hypothecation is still active on the car.

How does engine size affect the price of a car?
Larger engines usually increase the price because they offer stronger performance
and are often found in premium sedans, SUVs, or performance cars. Smaller engines
are more economical and common in budget-friendly cars. Engine size also affects
fuel economy, maintenance cost, and insurance, which influences both new-car and
used-car prices.

How do fuel type and running cost influence value?
Petrol cars are usually preferred for city use because they are smoother and often
have a lower initial cost. Diesel cars can suit long-distance users due to mileage
advantage. CNG cars appeal to cost-conscious buyers where infrastructure is
available. Electric vehicles have lower running costs but higher initial prices,
and their value depends on battery confidence and charging access.

How do ownership and service records affect resale?
Single-owner cars with complete authorised service history are generally easier to
trust and sell. Missing records, multiple owners, long service gaps, or visible
repair history reduce buyer confidence and can lower the final resale price.

How should the chatbot answer dataset-related car questions?
When the user asks about brands, resale, depreciation, mileage, fuel type, engine
size, used-car buying, or pricing in India, the assistant should answer confidently
using the domain context, explain the main factors clearly, and sound helpful and
knowledgeable rather than sounding like a retrieval system.

Tips for Buyers
---------------
1. Always get an independent inspection before buying a used car.
2. Check the Registration Certificate (RC) and verify it matches the seller's ID.
3. Confirm no outstanding loans (hypothecation) on the vehicle.
4. Prefer cars with remaining manufacturer warranty or extended warranty plans.
5. Compare prices across multiple platforms (Cars24, CarDekho, Spinny) before deciding.
6. Negotiate based on market value, not just the asking price.

Tips for Sellers
----------------
1. Keep full-service history documents ready.
2. A fresh service and cleaned interior/exterior can add 5–8% to resale value.
3. Price slightly above target to leave negotiation room.
4. Disclose any known issues honestly to avoid post-sale disputes.
5. Sell before the car crosses 1,00,000 km to maximise value.
"""

# ── 2. Chunk the knowledge base ───────────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
CHUNKS   = splitter.create_documents([KNOWLEDGE_BASE])

# ── 3. Build FAISS index (cached after first call) ───────────────────────────
_VECTORSTORE = None   # module-level cache


def _normalize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _keyword_overlap_score(query: str, text: str) -> int:
    query_tokens = set(_normalize(query))
    text_tokens = set(_normalize(text))
    return len(query_tokens & text_tokens)

def _get_vectorstore():
    global _VECTORSTORE
    if _VECTORSTORE is None:
        # Use a lightweight, CPU-friendly HuggingFace model
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )
        _VECTORSTORE = FAISS.from_documents(CHUNKS, embeddings)
    return _VECTORSTORE


def retrieve_docs(query: str, k: int = 4) -> str:
    """
    Retrieve the top-k most relevant knowledge base chunks for a query.

    Parameters
    ----------
    query : str   – the user's question
    k     : int   – number of chunks to retrieve (default 3)

    Returns
    -------
    str – concatenated text of retrieved chunks
    """
    vs = _get_vectorstore()
    semantic_results = vs.similarity_search(query, k=min(len(CHUNKS), 6))
    ranked_results = sorted(
        semantic_results,
        key=lambda doc: _keyword_overlap_score(query, doc.page_content),
        reverse=True,
    )
    selected = ranked_results[:k]
    context = "\n\n---\n\n".join(doc.page_content for doc in selected)
    return context


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ctx = retrieve_docs("How does fuel type affect car price?")
    print(ctx)
