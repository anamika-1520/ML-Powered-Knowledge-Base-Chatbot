"""
graph.py
--------
LangGraph chatbot that routes between:
  - ML node for car price prediction
  - RAG node for car knowledge questions
"""

import re
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph

from ml_tool import predict_car_price
from rag_faiss import retrieve_docs

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    max_tokens=2048,
)


class ChatState(TypedDict):
    user_message: str
    intent: str
    ml_result: dict | None
    rag_context: str | None
    direct_answer: str | None
    final_answer: str
    history: list[dict]


ML_KEYWORDS = [
    "price",
    "predict",
    "worth",
    "value",
    "estimate",
    "valuation",
    "quote",
]

EXPLANATION_HINTS = [
    "tell me about",
    "about",
    "distribution",
    "overall",
    "overview",
    "what",
    "why",
    "how",
    "explain",
    "difference",
    "compare",
    "factors",
]

CAR_KNOWLEDGE_HINTS = [
    "honda",
    "bmw",
    "toyota",
    "ford",
    "hyundai",
    "maruti",
    "suzuki",
    "maruti suzuki",
    "tata",
    "mahindra",
    "kia",
    "renault",
    "nissan",
    "skoda",
    "volkswagen",
    "audi",
    "mercedes",
    "mercedes-benz",
    "mg",
    "ferrari",
    "ferrai",
    "lamborghini",
    "porsche",
    "jaguar",
    "land rover",
    "range rover",
    "volvo",
    "lexus",
    "jeep",
    "citroen",
    "mini",
    "byd",
    "isuzu",
    "price",
    "pricing",
    "resale",
    "value",
    "worth",
    "depreciation",
    "mileage",
    "kilometre",
    "kilometer",
    "fuel",
    "service",
    "maintenance",
    "used car",
    "buying",
    "selling",
    "market",
    "car",
    "cars",
    "suv",
    "sedan",
    "hatchback",
    "ev",
]

RAG_TOOL_HINTS = [
    "rag tool",
    "using rag",
    "use rag",
    "via rag",
    "through rag",
]

SUPPORTED_BRANDS = ["Toyota", "Honda", "Ford", "BMW", "Hyundai", "Maruti"]
KNOWN_BRAND_ALIASES = {
    "ferrari": "Ferrari",
    "ferrai": "Ferrari",
    "lamborghini": "Lamborghini",
    "porsche": "Porsche",
    "jaguar": "Jaguar",
    "land rover": "Land Rover",
    "range rover": "Land Rover",
    "volvo": "Volvo",
    "lexus": "Lexus",
    "jeep": "Jeep",
    "citroen": "Citroen",
    "mini": "Mini",
    "byd": "BYD",
    "isuzu": "Isuzu",
    "mercedes": "Mercedes-Benz",
    "mercedes-benz": "Mercedes-Benz",
    "audi": "Audi",
    "skoda": "Skoda",
    "volkswagen": "Volkswagen",
    "tata": "Tata",
    "mahindra": "Mahindra",
    "kia": "Kia",
    "renault": "Renault",
    "nissan": "Nissan",
    "mg": "MG",
}

SMALL_TALK_PATTERNS = {
    "identity": [
        "who are you",
        "what are you",
        "tell me about yourself",
    ],
    "purpose": [
        "what is your task",
        "what is your purpose",
        "what do you do",
        "what can you do",
    ],
    "greeting": [
        "hi",
        "hii",
        "hiii",
        "hello",
        "hey",
        "good morning",
        "good evening",
    ],
    "chat": [
        "can you talk to me",
        "talk to me",
        "are you ok",
        "are u ok",
        "tell me something new",
        "do you know my name",
    ],
}

FAQ_DIRECT_ANSWERS = {
    "resale value": (
        "Car resale value in India depends on a mix of vehicle condition, brand perception, and market demand.\n\n"
        "Key factors:\n"
        "- Brand reputation: Brands with strong trust, service reach, and reliability usually retain value better.\n"
        "- Year of manufacture: Newer cars generally command higher resale because they feel more relevant and have lower age-related wear.\n"
        "- Kilometres driven: Lower usage suggests less wear and usually supports a better price.\n"
        "- Fuel type: Petrol, diesel, CNG, and EVs each attract different buyer groups depending on running cost and usage pattern.\n"
        "- Service history: A complete and clean service record increases buyer confidence.\n"
        "- Ownership history: Single-owner cars are often preferred over multi-owner cars.\n"
        "- Condition: Accident history, paint quality, tyres, interior wear, and mechanical health directly affect value.\n"
        "- Market demand: Popular body styles and brands sell faster and usually hold price better.\n\n"
        "In short, a trusted brand, lower kilometres, clean record, and good condition usually lead to the strongest resale value."
    ),
    "diesel vs petrol": (
        "Diesel cars usually cost more than petrol cars because diesel engines are built differently and are often designed for heavier, long-distance use.\n\n"
        "Main reasons:\n"
        "- Engine construction: Diesel engines are built to handle higher compression, which increases manufacturing cost.\n"
        "- Torque and highway use: Diesel cars are often preferred for long drives and heavy usage because they deliver strong torque and better highway efficiency.\n"
        "- Mileage advantage: Buyers who drive a lot may prefer diesel because lower running cost can offset the higher upfront price over time.\n"
        "- Segment positioning: Many diesel variants are offered in bigger cars, SUVs, or higher trims, which also pushes price upward.\n"
        "- Maintenance and regulation: Diesel ownership can involve different maintenance expectations and region-specific concerns, which affect perception and resale.\n\n"
        "So the higher price is not just about fuel type itself, but about engine design, intended usage, and market positioning."
    ),
    "mileage impact": (
        "Kilometres driven are one of the strongest factors in used-car pricing because they indicate how much wear the car may have experienced.\n\n"
        "How it affects price:\n"
        "- Low mileage: Usually increases value because buyers assume lower wear on engine, suspension, clutch, and tyres.\n"
        "- Average mileage: Gives a balanced price if the car is otherwise well maintained.\n"
        "- High mileage: Reduces value because buyers expect more maintenance, part replacement, and lower remaining life.\n"
        "- Usage pattern matters too: A well-maintained highway-driven car may sometimes be valued better than a poorly maintained city-driven car with lower kilometres.\n\n"
        "Typical interpretation:\n"
        "- Under 30,000 km: low usage\n"
        "- 30,000 to 80,000 km: average usage\n"
        "- Above 1,00,000 km: high usage\n\n"
        "Mileage does not work alone, but combined with service history and overall condition it has a major impact on resale price."
    ),
    "used car tips": (
        "Buying a used car in India is not just about finding a low price; it is about checking whether the car is genuine, well maintained, and fairly valued.\n\n"
        "Important checks:\n"
        "- Verify RC and ownership details to confirm the seller and vehicle match.\n"
        "- Check service history to understand how consistently the car was maintained.\n"
        "- Inspect for accident repair, repainting, rust, tyre wear, and interior condition.\n"
        "- Confirm insurance validity and claim history if available.\n"
        "- Make sure there is no active loan or hypothecation on the vehicle.\n"
        "- Take a proper test drive to evaluate engine smoothness, clutch, brakes, steering, and suspension.\n"
        "- Compare the asking price with similar listings on trusted platforms.\n"
        "- Prefer an independent inspection if you are unsure about condition.\n\n"
        "A good used car is not just cheap; it is mechanically sound, legally clear, and reasonably priced."
    ),
    "engine size": (
        "Engine size affects car price because it is closely linked to performance, segment, and ownership cost.\n\n"
        "Why bigger engines cost more:\n"
        "- Performance: Larger engines generally produce stronger power and torque.\n"
        "- Segment: Bigger engines are often found in premium sedans, SUVs, and performance-oriented cars.\n"
        "- Manufacturing and components: More powerful vehicles often come with stronger supporting systems and higher-spec equipment.\n"
        "- Running cost: Larger engines can increase fuel consumption, insurance, and maintenance expectations.\n\n"
        "Typical trend:\n"
        "- 800 to 1200 cc: economical, common in entry-level hatchbacks\n"
        "- 1200 to 2000 cc: balanced mix of performance and efficiency\n"
        "- Above 2000 cc: premium, SUV, or performance-oriented territory\n\n"
        "So engine size influences both the initial price and the long-term value perception of a car."
    ),
    "best car": (
        "There is no single best car for everyone, because the right choice depends on budget, usage, fuel preference, seating needs, and ownership priorities.\n\n"
        "A better way to judge the best car is by category:\n"
        "- Best for reliability and resale: brands like Toyota and Honda are often trusted.\n"
        "- Best for low running cost: Maruti Suzuki is commonly preferred.\n"
        "- Best for features and modern feel: Hyundai and Kia are often strong options.\n"
        "- Best for SUVs and road presence: Mahindra and Tata are commonly considered.\n"
        "- Best for luxury and performance: BMW, Audi, and Mercedes-Benz sit higher in the premium segment.\n\n"
        "So instead of one universal best car, the better answer is: the best car is the one that matches your budget, usage, and long-term ownership priorities."
    ),
    "maximum mileage": (
        "Cars that deliver the best mileage are usually smaller, lighter, and designed for efficiency rather than performance.\n\n"
        "General mileage logic:\n"
        "- Small petrol hatchbacks usually deliver strong everyday mileage.\n"
        "- Diesel cars can offer better highway efficiency, especially for long-distance users.\n"
        "- CNG cars are often chosen by buyers who want lower running cost where infrastructure is available.\n"
        "- Electric vehicles can offer the lowest running cost per kilometre, though their value depends on charging comfort and battery confidence.\n\n"
        "In practical ownership terms, buyers looking for maximum mileage often prefer brands and models known for efficiency, especially in Maruti Suzuki, Hyundai, and selected Tata or CNG-focused options.\n\n"
        "So the highest-mileage choice is usually not the most powerful car, but the one designed around fuel efficiency and low running cost."
    ),
    "best family car": (
        "A good family car is usually one that balances comfort, space, reliability, safety perception, and ownership cost.\n\n"
        "Important things families usually look for:\n"
        "- Comfortable cabin and rear-seat space\n"
        "- Good reliability and service support\n"
        "- Balanced resale value\n"
        "- Suitable seating capacity, often 5-seater or 7-seater depending on family size\n"
        "- Running cost that remains manageable over time\n\n"
        "In general, buyers often consider brands like Toyota, Honda, Hyundai, Maruti Suzuki, Tata, and Mahindra depending on whether they want sedan comfort, hatchback practicality, or SUV space.\n\n"
        "So the best family car is usually the one that gives comfort, reliability, and practical ownership rather than just high features or performance."
    ),
    "best city car": (
        "A good city car is usually compact, easy to drive, fuel-efficient, and comfortable in traffic.\n\n"
        "Key city-car qualities:\n"
        "- Easy maneuverability in traffic and tight parking\n"
        "- Good fuel efficiency for daily commuting\n"
        "- Smooth engine response at low speeds\n"
        "- Lower maintenance and running cost\n"
        "- Practical cabin space for regular daily use\n\n"
        "In this context, smaller hatchbacks and compact sedans from brands like Maruti Suzuki, Hyundai, Honda, and Tata are often considered suitable city-car choices.\n\n"
        "So the best city car is usually one that feels easy, efficient, and low-stress to own every day."
    ),
    "best resale": (
        "Cars with the best resale value are usually the ones that combine strong brand trust, high demand, reliable ownership image, and easier maintenance.\n\n"
        "What usually improves resale:\n"
        "- Popular and trusted brand reputation\n"
        "- Lower kilometres driven\n"
        "- Clean service and ownership history\n"
        "- Good overall condition\n"
        "- Strong demand in the used-car market\n\n"
        "In India, brands like Toyota, Honda, Maruti Suzuki, and selected Hyundai models are often associated with stronger resale compared with less popular or higher-maintenance alternatives.\n\n"
        "So the best resale usually comes from a car that is trusted, easy to maintain, and widely demanded in the used-car market."
    ),
    "worst car": (
        "There is no single worst car for everyone, because a car that feels weak for one buyer may still suit another buyer's budget or usage.\n\n"
        "Cars are usually judged poorly when they have one or more of these issues:\n"
        "- weak resale value\n"
        "- high maintenance relative to their segment\n"
        "- lower buyer trust in the used-car market\n"
        "- poor match between fuel cost and usage pattern\n"
        "- less practical space, comfort, or service support for the price\n\n"
        "So instead of asking for the worst car overall, it is better to ask: worst for resale, worst for mileage, worst for maintenance, or worst for family use. That gives a much more useful answer."
    ),
    "list of cars": (
        "Here are the main car brands and examples currently covered well in this chatbot.\n\n"
        "Core ML prediction brands:\n"
        "- Toyota\n"
        "- Honda\n"
        "- Ford\n"
        "- BMW\n"
        "- Hyundai\n"
        "- Maruti\n\n"
        "Broader RAG car-domain brands:\n"
        "- Tata\n"
        "- Mahindra\n"
        "- Kia\n"
        "- Renault\n"
        "- Nissan\n"
        "- Skoda\n"
        "- Volkswagen\n"
        "- Audi\n"
        "- Mercedes-Benz\n"
        "- MG\n"
        "- Ferrari\n"
        "- Lamborghini\n"
        "- Porsche\n"
        "- Jaguar\n"
        "- Land Rover\n"
        "- Volvo\n"
        "- Lexus\n"
        "- Jeep\n"
        "- Citroen\n"
        "- Mini\n"
        "- BYD\n"
        "- Isuzu\n\n"
        "If you want, I can also give you the ML-supported list only, or explain any one brand in detail."
    ),
    "ml tool cars": (
        "The ML prediction flow is currently built around these core dataset-supported brands:\n\n"
        "- Toyota\n"
        "- Honda\n"
        "- Ford\n"
        "- BMW\n"
        "- Hyundai\n"
        "- Maruti\n\n"
        "Example default model mapping used in the chatbot:\n"
        "- Toyota -> Corolla\n"
        "- Honda -> City\n"
        "- Ford -> EcoSport\n"
        "- BMW -> 3 Series\n"
        "- Hyundai -> Creta\n"
        "- Maruti -> Swift\n\n"
        "For price prediction, the chatbot mainly expects brand, model, year, kilometres driven, and fuel type."
    ),
    "car details": (
        "When evaluating any car properly, these are the main details that matter most:\n\n"
        "- Brand and model\n"
        "- Year of manufacture\n"
        "- Kilometres driven\n"
        "- Fuel type\n"
        "- Engine size and performance level\n"
        "- Seating capacity\n"
        "- Service history and ownership history\n"
        "- Condition, accident record, tyres, and interior quality\n"
        "- Resale value and running cost\n\n"
        "In short, good car evaluation is not just about brand name. It depends on practicality, maintenance, fuel cost, condition, and long-term ownership value."
    ),
}

BRAND_PROFILES = {
    "honda": {
        "opening": "Honda cars in India are generally known for refinement, reliable petrol engines, and comfortable everyday driving.",
        "positioning": "The brand is especially strong among buyers who want a practical family car with smooth city performance and balanced long-term ownership.",
        "strengths": [
            "Smooth petrol engine performance, especially for city and family use",
            "Comfortable cabin and practical daily-drive experience",
            "Strong reputation for reliability and balanced resale value",
            "Popular sedan choices like Honda City and Amaze, especially for comfort-focused buyers",
        ],
        "practicals": "Most mainstream Honda cars in this context are 5-seaters, with petrol being the strongest everyday-use association. Ownership cost is usually moderate rather than very cheap, but buyers often accept that for refinement and brand trust.",
        "closing": "Overall, Honda is usually seen as a dependable and refined choice for Indian buyers who want comfort, trust, and sensible ownership without entering premium-brand pricing.",
    },
    "bmw": {
        "opening": "BMW cars in India are associated with premium positioning, performance, and luxury ownership.",
        "positioning": "They are usually considered by buyers who want an executive car with strong road presence, sporty driving feel, and premium brand value.",
        "strengths": [
            "Strong brand image in the luxury segment",
            "Powerful engines and sporty driving dynamics",
            "Premium interiors, technology, and executive appeal",
            "Resale depends heavily on model year, kilometres driven, and service history",
        ],
        "practicals": "BMW cars in India are commonly 5-seaters in sedan and SUV segments. Petrol and diesel both appear in the luxury context, but overall ownership cost, maintenance, and insurance are significantly higher than mass-market brands.",
        "closing": "Overall, BMW is typically chosen by buyers who want luxury with performance character, but the total cost of ownership is also a major part of the decision.",
    },
    "toyota": {
        "opening": "Toyota cars in India are widely respected for reliability, durability, and strong long-term ownership value.",
        "positioning": "The brand is often preferred by buyers who want peace of mind, lower ownership stress, and dependable resale strength.",
        "strengths": [
            "Excellent reputation for mechanical reliability",
            "Strong resale value in the used-car market",
            "High buyer trust built around longevity and low-stress ownership",
            "Good appeal for families and practical long-term buyers",
        ],
        "practicals": "Toyota covers 5-seater and 7-seater family use in India depending on model category. Petrol and diesel both matter in Toyota discussions, and the brand often carries a slightly higher price because buyers trust its long-term durability.",
        "closing": "Overall, Toyota is usually seen as one of the safest brand choices for buyers who prioritise longevity, resale, and dependable ownership.",
    },
    "maruti suzuki": {
        "opening": "Maruti Suzuki cars are known in India for affordability, fuel efficiency, and easy ownership.",
        "positioning": "The brand is strongest among budget-conscious and practical buyers who want everyday usability with lower running stress.",
        "strengths": [
            "Lower running and maintenance costs than many rivals",
            "Very wide service network across India",
            "Strong popularity in hatchback and compact-car segments",
            "Easier resale because of buyer familiarity and practical ownership appeal",
        ],
        "practicals": "Most common Maruti offerings in this context are 5-seaters, though family-oriented larger options also exist. Petrol is strongly associated with Maruti ownership, and the brand is generally seen as one of the most cost-friendly choices in both purchase and running cost.",
        "closing": "Overall, Maruti Suzuki is usually the most practical choice for buyers who want simple ownership, lower maintenance pressure, and strong daily usability.",
    },
    "hyundai": {
        "opening": "Hyundai cars in India are known for being feature-rich, stylish, and value-oriented across multiple segments.",
        "positioning": "They usually appeal to buyers who want a polished mainstream car with modern design, comfort, and good equipment.",
        "strengths": [
            "Modern design and strong feature list",
            "Good balance of comfort, technology, and price",
            "Popular presence in hatchback, sedan, and SUV categories",
            "Strong appeal for buyers who want a polished all-round package",
        ],
        "practicals": "Hyundai is largely associated with 5-seater urban and family cars in hatchback, sedan, and SUV segments. Petrol and diesel both matter depending on model, and overall ownership cost usually sits in a balanced middle ground.",
        "closing": "Overall, Hyundai is usually seen as a smart mainstream brand for buyers who want modern features, comfort, and broad segment choice without going premium.",
    },
    "ford": {
        "opening": "Ford cars in India are often remembered for strong build quality, solid driving feel, and capable diesel performance.",
        "positioning": "They tend to appeal more to buyers who care about road feel, toughness, and driving confidence than pure low-cost ownership.",
        "strengths": [
            "Confident road feel and driver-focused character",
            "Good reputation for build quality in many models",
            "Strong appeal in selected used-car circles, especially for diesel buyers",
            "Value depends heavily on service support perception and local demand",
        ],
        "practicals": "Most mainstream Ford cars in this discussion are 5-seaters. Diesel has traditionally been a strong part of Ford's value perception, while total ownership appeal depends a lot on local service confidence and used-car buyer demand.",
        "closing": "Overall, Ford is often appreciated by buyers who care about driving feel and toughness, though resale can vary more than high-volume brands.",
    },
    "tata": {
        "opening": "Tata cars in India are increasingly associated with value, safety perception, and strong presence in both mainstream and EV segments.",
        "positioning": "The brand often attracts buyers who want visible value, road presence, and stronger safety appeal at a competitive price point.",
        "strengths": [
            "Competitive pricing with strong feature value",
            "Growing brand confidence among safety-focused buyers",
            "Strong visibility in hatchback, compact SUV, and EV categories",
            "Attractive option for buyers comparing practicality with modern features",
        ],
        "practicals": "Tata is mostly discussed in 5-seater hatchback and SUV contexts, with EV presence adding a different running-cost advantage. Petrol, diesel, and electric all matter depending on model, and overall ownership cost is often judged against the value and features offered.",
        "closing": "Overall, Tata is often seen as a value-driven brand with growing strength in Indian market relevance, especially in SUVs and electric vehicles.",
    },
    "mahindra": {
        "opening": "Mahindra cars in India are strongly associated with SUVs, rugged appeal, and strong road presence.",
        "positioning": "The brand is usually chosen by buyers who want a bold SUV identity, practical space, and a stronger utility-oriented image.",
        "strengths": [
            "Strong demand in SUV-focused segments",
            "Utility-oriented image with family and adventure appeal",
            "Popular diesel-driven and larger-vehicle reputation",
            "Resale often supported by demand for well-known SUV models",
        ],
        "practicals": "Mahindra is commonly linked with 5-seater and 7-seater SUVs in India. Diesel plays a strong role in its market identity, and overall ownership cost tends to be accepted by buyers who value space, presence, and SUV capability.",
        "closing": "Overall, Mahindra is usually seen as a strong SUV-first brand for buyers who want road presence, space, and robust appeal.",
    },
    "kia": {
        "opening": "Kia cars in India are known for modern styling, premium-looking interiors, and strong feature packaging.",
        "positioning": "The brand usually appeals to buyers who want a stylish, technology-focused mainstream vehicle with visible value.",
        "strengths": [
            "Attractive design and tech-focused appeal",
            "Competitive value in MPV and SUV segments",
            "Premium cabin feel relative to price point",
            "Strong appeal for buyers who want style and convenience features",
        ],
        "practicals": "Kia is often associated with 5-seater SUVs and family-oriented MPV usage depending on the model. Petrol and diesel both matter in its mainstream positioning, and buyers often judge it as a feature-rich option with balanced ownership appeal.",
        "closing": "Overall, Kia is often seen as a modern and feature-rich brand for buyers who want a more premium feel in the mainstream market.",
    },
    "renault": {
        "opening": "Renault cars in India are generally viewed as practical, budget-conscious options with value appeal in selected segments.",
        "positioning": "They usually attract buyers who want affordability and practicality more than premium feel or brand prestige.",
        "strengths": [
            "Good fit for entry-level and compact-car buyers",
            "Value-oriented ownership appeal",
            "Useful option in hatchback and compact SUV comparisons",
            "Demand depends strongly on model reputation and local buyer preference",
        ],
        "practicals": "Renault is commonly discussed in 5-seater hatchback and compact SUV contexts. Petrol is often the easier everyday-use association, and overall ownership appeal depends heavily on how buyers rate the specific model and service access.",
        "closing": "Overall, Renault is often considered by buyers who want affordability and practicality in a smaller or budget-focused vehicle.",
    },
    "nissan": {
        "opening": "Nissan cars in India are usually discussed in terms of selected practical models and value-conscious ownership.",
        "positioning": "The brand tends to be considered when buyers are exploring alternatives beyond the highest-volume mainstream names.",
        "strengths": [
            "Reasonable value positioning in selected segments",
            "Buyer interest often tied closely to model reputation",
            "Ownership appeal depends on local service access and demand",
            "Can be relevant for budget-aware buyers comparing fewer mainstream options",
        ],
        "practicals": "Nissan is generally associated with 5-seater practical-use models in this context. Fuel-type preference depends on model, while overall ownership value is judged more by local support and buyer confidence than pure brand pull.",
        "closing": "Overall, Nissan is generally a niche but practical consideration where the specific model and local support matter a lot.",
    },
    "skoda": {
        "opening": "Skoda cars in India are known for solid build quality, mature styling, and a more premium mainstream experience.",
        "positioning": "They usually appeal to buyers who want a refined, well-built car with a slightly upscale personality without moving into full luxury pricing.",
        "strengths": [
            "Strong build and stable road manners",
            "Premium feel without entering full luxury pricing",
            "Good appeal in sedan and SUV categories",
            "Attracts buyers who want comfort, quality, and a more mature driving experience",
        ],
        "practicals": "Skoda is commonly discussed in 5-seater sedan and SUV segments. Petrol is a strong current everyday association, and ownership cost is usually seen as higher than budget brands but acceptable for the premium-mainstream feel.",
        "closing": "Overall, Skoda is usually chosen by buyers who want quality, comfort, and a more refined ownership experience than typical mass-market options.",
    },
    "volkswagen": {
        "opening": "Volkswagen cars in India are often appreciated for strong build quality, stability, and a European-style driving feel.",
        "positioning": "They usually attract buyers who care more about engineering feel, stability, and mature road manners than flashy mass-market appeal.",
        "strengths": [
            "Good road manners and mature driving experience",
            "Solid build perception in mainstream segments",
            "Strong appeal for buyers who prioritise feel over flashy features",
            "Often compared with Skoda and Hyundai in value-versus-premium discussions",
        ],
        "practicals": "Volkswagen is usually associated with 5-seater hatchback, sedan, and SUV usage in India. Petrol plays an important role in current perception, while total ownership cost is often judged as premium-mainstream rather than low-cost.",
        "closing": "Overall, Volkswagen is usually seen as a smart choice for buyers who want a well-engineered, stable, and more driver-oriented car.",
    },
    "audi": {
        "opening": "Audi cars in India represent premium luxury, advanced features, and strong executive presence.",
        "positioning": "They usually attract buyers who want prestige, technology, and premium comfort in sedan and SUV form.",
        "strengths": [
            "Luxury positioning with premium interiors and technology",
            "Strong brand value in high-end sedan and SUV categories",
            "Pricing influenced by imported components and maintenance expectations",
            "Resale depends heavily on model desirability, condition, and service record",
        ],
        "practicals": "Audi cars in this context are mostly 5-seater premium sedans and SUVs. Petrol and diesel can both matter depending on model generation, but overall cost remains firmly on the expensive side in purchase, maintenance, and insurance.",
        "closing": "Overall, Audi is usually chosen by buyers who want premium comfort, prestige, and technology, with ownership cost being an important consideration.",
    },
    "mercedes-benz": {
        "opening": "Mercedes-Benz cars in India are associated with luxury, comfort, prestige, and premium ownership.",
        "positioning": "The brand usually appeals to buyers who prioritise comfort, status, and a refined premium experience more than pure budget efficiency.",
        "strengths": [
            "Strong brand prestige in the luxury market",
            "Comfort-oriented premium driving experience",
            "High-end interiors, features, and executive appeal",
            "Resale depends on age, kilometres driven, condition, and service history",
        ],
        "practicals": "Mercedes-Benz is commonly discussed in 5-seater luxury sedans and SUVs. Petrol and diesel both appear in its premium portfolio, but total ownership cost is high due to pricing, maintenance, insurance, and luxury positioning.",
        "closing": "Overall, Mercedes-Benz is typically seen as a benchmark luxury choice for buyers who prioritise comfort, status, and refined premium ownership.",
    },
    "mg": {
        "opening": "MG cars in India are known for feature-loaded SUVs, spacious cabins, and a modern ownership image.",
        "positioning": "They often appeal to buyers who want visible value, technology features, and a more modern feel in selected segments.",
        "strengths": [
            "Strong feature value and technology appeal",
            "Noticeable presence in SUV and EV-related discussions",
            "Good appeal for buyers who want a modern, spacious package",
            "Brand perception is often linked to convenience and visible value",
        ],
        "practicals": "MG is commonly associated with 5-seater SUV-style usage in India, with EV presence adding a different running-cost angle. Overall ownership perception is usually tied to features, space, and technology-led value.",
        "closing": "Overall, MG is usually seen as a brand for buyers who want a tech-friendly and feature-rich vehicle in selected high-interest segments.",
    },
    "ferrari": {
        "opening": "Ferrari cars are associated with ultra-premium performance, exclusivity, and exotic sports-car ownership.",
        "positioning": "The brand usually appeals to enthusiasts, collectors, and buyers who value heritage, performance, and status far above everyday practicality.",
        "strengths": [
            "Extremely strong luxury and performance brand image",
            "High-performance engines and sports-car character",
            "Exclusive ownership appeal with strong collector value in niche cases",
            "Pricing and resale depend heavily on model rarity, condition, and service history",
        ],
        "practicals": "Ferrari cars are typically low-volume premium imports and are not part of mainstream family-car ownership in India. They are usually 2-seater or 4-seat grand-tourer style offerings depending on model, with very high purchase cost, maintenance cost, and insurance expectations.",
        "closing": "Overall, Ferrari is usually seen as an aspirational exotic brand where exclusivity, performance, and prestige matter much more than everyday running-cost efficiency.",
    },
    "lamborghini": {
        "opening": "Lamborghini cars are associated with extreme styling, exotic performance, and ultra-luxury ownership.",
        "positioning": "The brand usually appeals to buyers who want dramatic road presence, high-end performance, and a strong statement of status and exclusivity.",
        "strengths": [
            "Very strong supercar identity and visual presence",
            "High-performance engines and premium engineering",
            "Ultra-luxury ownership appeal in a niche market",
            "Resale depends on rarity, condition, and collector-driven demand",
        ],
        "practicals": "Lamborghini ownership in India sits far outside mainstream car buying. Most models are exotic 2-seater supercars or premium performance SUVs, with extremely high purchase, maintenance, and insurance costs.",
        "closing": "Overall, Lamborghini is usually seen as a prestige-driven exotic performance brand for buyers who prioritise presence, exclusivity, and extreme design over everyday practicality.",
    },
    "porsche": {
        "opening": "Porsche cars are known for premium performance, strong engineering reputation, and a more driver-focused luxury identity.",
        "positioning": "The brand usually attracts buyers who want a balance of luxury, performance, and engineering depth rather than only badge prestige.",
        "strengths": [
            "Strong reputation for performance engineering",
            "Premium sporty character across coupe, sedan, and SUV categories",
            "High desirability among enthusiasts and premium buyers",
            "Resale depends on model appeal, condition, and maintenance history",
        ],
        "practicals": "Porsche in India is generally associated with premium imports, often in 4-seater sports or 5-seater SUV contexts depending on model. Overall ownership cost is high, but buyers often justify it through performance quality and brand reputation.",
        "closing": "Overall, Porsche is usually seen as a premium performance brand for buyers who want luxury with deeper driver appeal and engineering credibility.",
    },
    "jaguar": {
        "opening": "Jaguar cars are associated with premium luxury, elegant styling, and a refined executive image.",
        "positioning": "The brand generally appeals to buyers who want a luxury car with distinctive styling and a more niche premium identity.",
        "strengths": [
            "Luxury positioning with refined design appeal",
            "Executive sedan and premium SUV presence",
            "Distinctive brand identity compared with more common luxury rivals",
            "Value depends strongly on model condition, service history, and buyer demand",
        ],
        "practicals": "Jaguar ownership in India is mostly in premium 5-seater sedan and SUV contexts. Petrol and diesel both matter depending on generation and model, while total ownership cost is firmly premium in pricing, maintenance, and insurance.",
        "closing": "Overall, Jaguar is usually seen as a stylish and niche luxury brand for buyers who want premium comfort with a more distinctive identity.",
    },
    "land rover": {
        "opening": "Land Rover cars are associated with premium SUVs, strong road presence, and luxury adventure-oriented ownership.",
        "positioning": "The brand usually appeals to buyers who want a luxury SUV image with space, capability, and prestige.",
        "strengths": [
            "Strong premium SUV identity",
            "High brand appeal in luxury off-road and road-presence discussions",
            "Spacious and prestige-oriented ownership experience",
            "Resale depends on model desirability, upkeep, and premium buyer demand",
        ],
        "practicals": "Land Rover is typically discussed in 5-seater and 7-seater luxury SUV contexts in India. Ownership cost is high across purchase, maintenance, and insurance, and the brand sits firmly in the premium imported-SUV category.",
        "closing": "Overall, Land Rover is usually seen as a premium SUV brand for buyers who want prestige, presence, and luxury-utility appeal together.",
    },
    "volvo": {
        "opening": "Volvo cars are associated with safety, premium comfort, and understated luxury.",
        "positioning": "The brand generally appeals to buyers who want a premium car with strong safety perception and a more understated image than flashier luxury rivals.",
        "strengths": [
            "Strong safety-oriented brand identity",
            "Premium comfort and refined driving experience",
            "Appeal for buyers who value calm luxury over aggressive brand show",
            "Resale depends on model acceptance, condition, and premium buyer interest",
        ],
        "practicals": "Volvo in India is often discussed in 5-seater premium sedan and SUV contexts. Overall cost of ownership is premium, but buyers often associate the brand with sensible luxury, safety, and comfort rather than pure performance.",
        "closing": "Overall, Volvo is usually seen as a thoughtful premium brand for buyers who prioritise safety, comfort, and understated luxury.",
    },
    "lexus": {
        "opening": "Lexus cars are associated with premium comfort, refinement, and high-end ownership with a strong reliability image.",
        "positioning": "The brand usually appeals to buyers who want luxury with smoothness, comfort, and a calmer premium experience rather than aggressive sportiness.",
        "strengths": [
            "Premium comfort-focused ownership image",
            "Strong association with refinement and reliability",
            "High-end interiors and smooth driving character",
            "Resale depends on rarity, upkeep, and premium buyer demand",
        ],
        "practicals": "Lexus in India is usually seen in premium sedan and SUV contexts, often with 5-seat luxury orientation. Total ownership cost is high, but the brand is often judged positively for comfort, refinement, and premium calmness.",
        "closing": "Overall, Lexus is usually seen as a refined luxury brand for buyers who want premium comfort and smooth ownership character with a strong trust image.",
    },
    "jeep": {
        "opening": "Jeep cars are associated with SUVs, rugged image, and premium off-road-oriented appeal.",
        "positioning": "The brand generally appeals to buyers who want an SUV with stronger adventure identity and more distinct road presence than standard urban crossovers.",
        "strengths": [
            "Strong SUV and rugged-brand perception",
            "Good appeal among buyers who value road presence and capability image",
            "Distinctive identity compared with generic mainstream SUVs",
            "Value depends on model demand, upkeep, and buyer perception",
        ],
        "practicals": "Jeep in India is usually discussed in 5-seater SUV contexts. Petrol and diesel can both matter depending on model, while ownership cost is typically higher than mainstream SUV brands but lower than top-end luxury imports.",
        "closing": "Overall, Jeep is usually seen as a more premium SUV choice for buyers who want rugged identity, presence, and capability-oriented brand appeal.",
    },
    "citroen": {
        "opening": "Citroen cars are generally associated with a niche mainstream identity, comfort-oriented appeal, and selected value-focused offerings.",
        "positioning": "The brand usually attracts buyers who are open to alternatives beyond the biggest mainstream names and want something slightly different in design or character.",
        "strengths": [
            "Distinct identity compared with mass-market rivals",
            "Potential comfort-oriented appeal in selected models",
            "Useful as an alternative choice in compact segments",
            "Demand depends strongly on model reputation and market acceptance",
        ],
        "practicals": "Citroen in India is mostly discussed in 5-seater hatchback or compact-SUV style contexts. Ownership appeal depends a lot on local service confidence, buyer trust, and the specific model rather than only brand pull.",
        "closing": "Overall, Citroen is usually seen as a niche alternative brand for buyers who want something different from the standard mainstream choices.",
    },
    "mini": {
        "opening": "Mini cars are associated with premium compact styling, urban character, and distinctive design-led ownership.",
        "positioning": "The brand usually appeals to buyers who want a small premium car with strong personality and lifestyle value rather than pure space efficiency.",
        "strengths": [
            "Distinctive premium compact design identity",
            "Strong urban and lifestyle appeal",
            "Premium ownership character in a small-car format",
            "Value depends on rarity, condition, and niche buyer demand",
        ],
        "practicals": "Mini in India is typically a premium compact 4-seat or 5-seat urban-oriented choice depending on body style. Ownership cost is premium relative to size, so it is bought more for personality and badge appeal than practicality alone.",
        "closing": "Overall, Mini is usually seen as a premium lifestyle brand for buyers who want compact size with strong design character and brand identity.",
    },
    "byd": {
        "opening": "BYD cars are associated with electric mobility, modern technology, and growing EV-focused interest.",
        "positioning": "The brand usually appeals to buyers who are specifically exploring electric-car ownership and want technology-led value in a newer market space.",
        "strengths": [
            "Strong EV-focused brand perception",
            "Appeal driven by battery-electric positioning and technology value",
            "Relevant in conversations about future mobility and lower running cost potential",
            "Market value depends on EV confidence, charging comfort, and buyer acceptance",
        ],
        "practicals": "BYD in India is mainly discussed in EV contexts, often with premium or upper-mainstream positioning depending on model. Running-cost potential can be attractive, but total ownership perception depends heavily on charging confidence, battery trust, and market adoption.",
        "closing": "Overall, BYD is usually seen as an EV-focused brand for buyers who want modern electric-car value and are comfortable with a newer-brand ownership journey.",
    },
    "isuzu": {
        "opening": "Isuzu cars are associated with utility-focused vehicles, rugged capability, and pickup or large-vehicle practicality.",
        "positioning": "The brand usually appeals to buyers who value toughness, utility use, and work-oriented capability more than mainstream family-car softness.",
        "strengths": [
            "Strong utility and rugged-vehicle identity",
            "Appeal in pickup and practical heavy-duty discussions",
            "Useful reputation for toughness and purpose-driven ownership",
            "Demand depends on specific use case, condition, and local buyer interest",
        ],
        "practicals": "Isuzu in India is more often linked with utility-oriented larger vehicles than standard urban passenger-car ownership. Diesel plays an important role in its image, and overall value depends strongly on use case and buyer expectation.",
        "closing": "Overall, Isuzu is usually seen as a practical rugged brand for buyers who want capability, durability, and utility-focused ownership.",
    },
}


def _format_brand_answer(profile: dict) -> str:
    strengths = "\n".join(f"• {point}" for point in profile["strengths"])
    return (
        f"{profile['opening']}\n"
        f"{profile['positioning']}\n\n"
        "Key strengths:\n"
        f"{strengths}\n\n"
        "Ownership and practical view:\n"
        f"{profile['practicals']}\n\n"
        f"{profile['closing']}"
    )


def _extract_features(text: str) -> dict:
    """Extract features compatible with the saved ML model."""
    features = {}

    km_match = re.search(r"(\d[\d,]*)\s*(?:km|kms|kilometers|kilometres)", text, re.I)
    if km_match:
        features["kms_driven"] = int(km_match.group(1).replace(",", ""))

    year_matches = list(re.finditer(r"\b(19\d{2}|20[0-2][0-9])\b", text))
    for year_match in year_matches:
        # Do not treat the number inside "2000 km" as a manufacturing year.
        if km_match and year_match.span() == km_match.span(1):
            continue
        features["year"] = int(year_match.group(1))
        break

    for brand in SUPPORTED_BRANDS:
        if brand.lower() in text.lower():
            features["company"] = brand
            break
    if "company" not in features:
        lowered = text.lower()
        for alias, canonical in KNOWN_BRAND_ALIASES.items():
            if alias in lowered:
                features["company"] = canonical
                break

    for fuel in ["Petrol", "Diesel", "CNG", "Electric"]:
        if fuel.lower() in text.lower():
            features["fuel_type"] = fuel
            break

    model_patterns = [
        r"\bBMW\s+(?!with\b)([A-Za-z0-9-]+(?:\s+[A-Za-z0-9-]+)?)",
        r"\bHonda\s+(?!with\b)([A-Za-z0-9-]+(?:\s+[A-Za-z0-9-]+)?)",
        r"\bToyota\s+(?!with\b)([A-Za-z0-9-]+(?:\s+[A-Za-z0-9-]+)?)",
        r"\bFord\s+(?!with\b)([A-Za-z0-9-]+(?:\s+[A-Za-z0-9-]+)?)",
        r"\bHyundai\s+(?!with\b)([A-Za-z0-9-]+(?:\s+[A-Za-z0-9-]+)?)",
        r"\bMaruti\s+(?!with\b)([A-Za-z0-9-]+(?:\s+[A-Za-z0-9-]+)?)",
    ]
    for pattern in model_patterns:
        model_match = re.search(pattern, text, re.I)
        if not model_match:
            continue
        candidate = model_match.group(1).strip(" ,.")
        if candidate and not re.fullmatch(
            r"\d{4}|\d[\d,]*|Petrol|Diesel|CNG|Electric|\d+\s*Seats?", candidate, re.I
        ):
            features["name"] = candidate.title()
            break

    if "name" not in features and "company" in features:
        company_defaults = {
            "BMW": "3 Series",
            "Honda": "City",
            "Toyota": "Corolla",
            "Ford": "EcoSport",
            "Hyundai": "Creta",
            "Maruti": "Swift",
            "Mercedes-Benz": "C-Class",
            "Audi": "A4",
            "Skoda": "Slavia",
            "Volkswagen": "Virtus",
            "Tata": "Nexon",
            "Mahindra": "XUV700",
            "Kia": "Seltos",
            "Renault": "Kiger",
            "Nissan": "Magnite",
            "MG": "Hector",
            "Ferrari": "Portofino",
            "Lamborghini": "Urus",
        }
        features["name"] = company_defaults.get(features["company"], "City")

    return features


def _llm_answer(system_prompt: str, user_message: str, context: str) -> str:
    prompt = f"Context:\n{context}\n\nUser: {user_message}"

    try:
        response = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt),
            ]
        )
        answer = response.content.strip()
        if not answer or len(answer) < 10:
            # Fallback: synthesize a basic answer
            answer = (
                "Based on car market knowledge, Honda cars in India are known for reliability and good resale value. "
                "For specific pricing, factors like model, year, kilometers, and fuel type matter. "
                "Generally, used Honda City or Amaze can range from ₹4-10 lakhs depending on condition. "
                "For exact estimates, try the ML tool with full details."
            )
        return answer
    except Exception:
        return (
            "Sorry, I couldn't process that right now. "
            "Honda is a trusted brand for practical cars in India. "
            "For pricing, provide model details like year and kilometers."
        )


def _missing_ml_fields(features: dict) -> list[str]:
    missing = []
    if "company" not in features:
        missing.append("company")
    if "year" not in features:
        missing.append("year")
    if "kms_driven" not in features:
        missing.append("kilometres driven")
    if "fuel_type" not in features:
        missing.append("fuel type")
    return missing


def _extract_brands(text: str) -> list[str]:
    found = []
    lowered = text.lower()
    for brand in SUPPORTED_BRANDS:
        if brand.lower() in lowered:
            found.append(brand)
    for alias, canonical in KNOWN_BRAND_ALIASES.items():
        if alias in lowered and canonical not in found:
            found.append(canonical)
    return found


def _is_in_kb_domain(message: str) -> bool:
    msg = message.lower()
    return any(hint in msg for hint in CAR_KNOWLEDGE_HINTS)


def _build_ml_clarification(features: dict) -> str:
    missing = _missing_ml_fields(features)
    brand = features.get("company", "this car")
    missing_text = ", ".join(missing)
    example_by_brand = {
        "BMW": "Predict price of 2021 BMW 3 Series Petrol 35,000 km",
        "Honda": "Predict price of 2021 Honda City Petrol 35,000 km",
        "Toyota": "Predict price of 2020 Toyota Corolla Petrol 40,000 km",
        "Ford": "Predict price of 2020 Ford EcoSport Diesel 45,000 km",
        "Hyundai": "Predict price of 2022 Hyundai Creta Petrol 20,000 km",
        "Maruti": "Predict price of 2021 Maruti Swift Petrol 25,000 km",
        "Mercedes-Benz": "Predict price of 2021 Mercedes-Benz C-Class Petrol 22,000 km",
        "Audi": "Predict price of 2020 Audi A4 Petrol 30,000 km",
        "Skoda": "Predict price of 2022 Skoda Slavia Petrol 18,000 km",
        "Volkswagen": "Predict price of 2022 Volkswagen Virtus Petrol 18,000 km",
        "Tata": "Predict price of 2023 Tata Nexon Petrol 15,000 km",
        "Mahindra": "Predict price of 2022 Mahindra XUV700 Diesel 25,000 km",
        "Kia": "Predict price of 2022 Kia Seltos Petrol 20,000 km",
        "Renault": "Predict price of 2021 Renault Kiger Petrol 28,000 km",
        "Nissan": "Predict price of 2021 Nissan Magnite Petrol 30,000 km",
        "MG": "Predict price of 2022 MG Hector Petrol 22,000 km",
        "Ferrari": "Predict price of 2021 Ferrari Portofino Petrol 8,000 km",
        "Lamborghini": "Predict price of 2021 Lamborghini Urus Petrol 10,000 km",
        "Porsche": "Predict price of 2021 Porsche Cayenne Petrol 18,000 km",
        "Jaguar": "Predict price of 2020 Jaguar XF Petrol 24,000 km",
        "Land Rover": "Predict price of 2021 Land Rover Discovery Sport Diesel 20,000 km",
        "Volvo": "Predict price of 2021 Volvo XC60 Petrol 18,000 km",
        "Lexus": "Predict price of 2021 Lexus ES Petrol 20,000 km",
        "Jeep": "Predict price of 2021 Jeep Compass Diesel 28,000 km",
        "Citroen": "Predict price of 2022 Citroen C3 Petrol 16,000 km",
        "Mini": "Predict price of 2021 Mini Cooper Petrol 14,000 km",
        "BYD": "Predict price of 2023 BYD Atto 3 Electric 12,000 km",
        "Isuzu": "Predict price of 2021 Isuzu D-Max Diesel 35,000 km",
    }
    example_text = example_by_brand.get(brand, "Predict price of 2021 BMW 3 Series Petrol 35,000 km")
    return (
        f"To estimate the price of {brand}, I need a few more details.\n\n"
        "Please share: model name, year, kilometres driven, and fuel type.\n\n"
        "Example:\n"
        f"{example_text}"
    )


def _build_multi_car_answer(brands: list[str]) -> str:
    joined = " and ".join(brands[:2]) if len(brands) == 2 else ", ".join(brands)
    return (
        f"I can estimate one car at a time, but I detected multiple brands: {joined}.\n\n"
        "Please send each car separately, or provide full details for one car first.\n\n"
        "Examples:\n"
        "- Predict price of 2021 Honda City Petrol 35,000 km\n"
        "- Predict price of 2020 Maruti Swift Petrol 28,000 km"
    )


def _build_out_of_scope_answer() -> str:
    return (
        "I do not have reliable data for that in this chatbot yet.\n"
        "Right now, the knowledge base is focused on car pricing, resale factors, "
        "depreciation, fuel type impact, and used-car market basics in India."
    )


def _wants_rag_tool(message: str) -> bool:
    msg = message.lower()
    return any(hint in msg for hint in RAG_TOOL_HINTS)


def _small_talk_answer(message: str) -> str | None:
    msg = message.lower().strip()

    if any(pattern in msg for pattern in SMALL_TALK_PATTERNS["identity"]):
        return (
            "I am AutoMate, a car-price and car-knowledge chatbot.\n\n"
            "I can help with:\n"
            "- car price prediction\n"
            "- car resale factors\n"
            "- fuel type, mileage, and depreciation questions\n"
            "- used-car buying guidance in India"
        )

    if any(pattern in msg for pattern in SMALL_TALK_PATTERNS["purpose"]):
        return (
            "My task is to help with Indian car-related questions.\n\n"
            "I can:\n"
            "- predict used-car prices from details like brand, year, fuel, and kilometres driven\n"
            "- answer car pricing and resale questions using my knowledge base\n"
            "- explain factors like mileage, engine size, fuel type, and service history"
        )

    if msg in SMALL_TALK_PATTERNS["greeting"] or any(
        msg.startswith(pattern + " ") for pattern in SMALL_TALK_PATTERNS["greeting"]
    ):
        return (
            "Hi! I am AutoMate 🚗\n\n"
            "You can ask me about car prices, resale value, mileage impact, fuel type, "
            "or used-car buying tips in India."
        )

    if "do you know my name" in msg:
        return (
            "I only know what you share with me in this chat.\n\n"
            "If you want, you can tell me your name and I will reply accordingly in this conversation."
        )

    if "can you talk to me" in msg or "talk to me" in msg:
        return (
            "Yes, I can talk to you.\n\n"
            "I am mainly built for car-related help, but I can also handle simple conversation and then guide you back to car questions when needed."
        )

    if "are you ok" in msg or "are u ok" in msg:
        return "Yes, I am here and ready to help. You can ask me about cars or start with any simple question."

    if "tell me something new" in msg:
        return (
            "Here is one useful car fact:\n\n"
            "In used-car pricing, low kilometres driven and a clean service history often matter almost as much as the brand itself.\n\n"
            "If you want, I can also tell you something new about resale value, fuel type, or car buying."
        )

    return None


def _faq_direct_answer(message: str) -> str | None:
    msg = message.lower()

    if "resale value" in msg or "affect car resale" in msg:
        return FAQ_DIRECT_ANSWERS["resale value"]
    if "diesel cars cost more than petrol" in msg or "diesel cost more than petrol" in msg:
        return FAQ_DIRECT_ANSWERS["diesel vs petrol"]
    if "mileage" in msg or "km driven" in msg or "kilometres driven" in msg:
        return FAQ_DIRECT_ANSWERS["mileage impact"]
    if "tips for buying a used car" in msg or "buying a used car" in msg:
        return FAQ_DIRECT_ANSWERS["used car tips"]
    if "engine size affect" in msg or "engine size" in msg:
        return FAQ_DIRECT_ANSWERS["engine size"]
    if "best car" in msg or "sabse achhi car" in msg:
        return FAQ_DIRECT_ANSWERS["best car"]
    if "maximum mileage" in msg or "highest mileage" in msg or "sabse jyada mileage" in msg:
        return FAQ_DIRECT_ANSWERS["maximum mileage"]
    if "best family car" in msg or "family car" in msg:
        return FAQ_DIRECT_ANSWERS["best family car"]
    if "best city car" in msg or "city car" in msg:
        return FAQ_DIRECT_ANSWERS["best city car"]
    if "best resale" in msg or "highest resale" in msg:
        return FAQ_DIRECT_ANSWERS["best resale"]
    if "worst car" in msg or "sabse bekar car" in msg:
        return FAQ_DIRECT_ANSWERS["worst car"]
    if (
        "list of cars in ml tool" in msg
        or "cars in ml tool" in msg
        or "ml prediction dataset cars" in msg
        or "ml tool list" in msg
        or "list of cars using ml tool" in msg
    ):
        return FAQ_DIRECT_ANSWERS["ml tool cars"]
    if (
        "list of cars" in msg
        or "give list of cars" in msg
        or "cars list" in msg
        or ("list" in msg and "cars" in msg)
        or ("give" in msg and "list" in msg and "cars" in msg)
    ):
        return FAQ_DIRECT_ANSWERS["list of cars"]
    if (
        "car details" in msg
        or "details about car" in msg
        or "details of car" in msg
        or "car related details" in msg
    ):
        return FAQ_DIRECT_ANSWERS["car details"]

    return None


def _brand_direct_answer(message: str) -> str | None:
    msg = message.lower()

    ordered_brands = [
        "land rover",
        "mercedes-benz",
        "maruti suzuki",
        "volkswagen",
        "lamborghini",
        "ferrari",
        "porsche",
        "jaguar",
        "volvo",
        "lexus",
        "jeep",
        "citroen",
        "mini",
        "byd",
        "isuzu",
        "hyundai",
        "mahindra",
        "renault",
        "nissan",
        "toyota",
        "honda",
        "bmw",
        "tata",
        "ford",
        "kia",
        "skoda",
        "audi",
        "mg",
    ]
    for brand in ordered_brands:
        if brand in msg:
            return _format_brand_answer(BRAND_PROFILES[brand])

    if "maruti" in msg or "suzuki" in msg:
        return _format_brand_answer(BRAND_PROFILES["maruti suzuki"])
    if "mercedes" in msg:
        return _format_brand_answer(BRAND_PROFILES["mercedes-benz"])
    if "range rover" in msg:
        return _format_brand_answer(BRAND_PROFILES["land rover"])

    return None


def input_node(state: ChatState) -> ChatState:
    return state


def router_node(state: ChatState) -> ChatState:
    msg = state["user_message"].lower().strip()
    extracted = _extract_features(state["user_message"])
    detected_brands = _extract_brands(state["user_message"])
    small_talk = _small_talk_answer(state["user_message"])
    faq_answer = _faq_direct_answer(state["user_message"])
    brand_answer = _brand_direct_answer(state["user_message"])

    has_prediction_keyword = any(keyword in msg for keyword in ML_KEYWORDS)
    has_explanation_hint = any(hint in msg for hint in EXPLANATION_HINTS)
    force_rag = _wants_rag_tool(state["user_message"])
    feature_count = sum(
        key in extracted for key in ["company", "year", "kms_driven", "fuel_type"]
    )
    has_enough_ml_details = feature_count >= 3 and "year" in extracted

    intent = "rag"
    direct_answer = None

    if small_talk:
        intent = "direct"
        direct_answer = small_talk
    elif faq_answer:
        intent = "direct"
        direct_answer = faq_answer
    elif force_rag:
        intent = "rag"
    elif brand_answer and not has_prediction_keyword:
        intent = "direct"
        direct_answer = brand_answer
    elif has_prediction_keyword and not has_explanation_hint:
        if len(detected_brands) > 1:
            intent = "direct"
            direct_answer = _build_multi_car_answer(detected_brands)
        elif has_enough_ml_details:
            intent = "ml"
        else:
            intent = "direct"
            direct_answer = _build_ml_clarification(extracted)
    elif not _is_in_kb_domain(state["user_message"]):
        intent = "direct"
        direct_answer = _build_out_of_scope_answer()

    print(f"DEBUG intent: {intent}")
    return {**state, "intent": intent, "direct_answer": direct_answer}


def ml_node(state: ChatState) -> ChatState:
    print("DEBUG ML NODE CALLED")
    features = _extract_features(state["user_message"])
    ml_result = predict_car_price(features)
    return {**state, "ml_result": ml_result}


def rag_node(state: ChatState) -> ChatState:
    print("DEBUG RAG NODE CALLED")
    context = retrieve_docs(state["user_message"])
    return {**state, "rag_context": context}


def direct_node(state: ChatState) -> ChatState:
    print("DEBUG DIRECT NODE CALLED")
    return state


def response_node(state: ChatState) -> ChatState:
    if state["intent"] == "direct" and state.get("direct_answer"):
        answer = state["direct_answer"]
    elif state["intent"] == "ml" and state.get("ml_result"):
        result = state["ml_result"]
        price = result["predicted_price"]
        conf = result["confidence_note"]
        used = result["input_used"]

        if price is None:
            answer = (
                "I could not generate a car-price prediction.\n\n"
                "Please provide:\n"
                "- company\n"
                "- year\n"
                "- kilometres driven\n"
                "- fuel type"
            )
        else:
            lower_bound = max(10000, round(price * 0.95, -3))
            upper_bound = round(price * 1.05, -3)
            answer = (
                "Estimated Used-Car Price\n"
                f"Rs. {price:,.0f}\n\n"
                "Expected Range\n"
                f"Rs. {lower_bound:,.0f} - Rs. {upper_bound:,.0f}\n\n"
                f"Confidence\n{conf}\n\n"
                "Details Used\n"
                f"- Company: {used.get('company')}\n"
                f"- Model: {used.get('name')}\n"
                f"- Year: {used.get('year')}\n"
                f"- KM Driven: {used.get('kms_driven')}\n"
                f"- Fuel: {used.get('fuel_type')}\n"
                "\nNote\n"
                "Final resale value can vary by city, condition, ownership history, and service record."
            )
    else:
        context = state.get("rag_context", "")
        system = (
            "You are a helpful car-domain assistant for an assignment chatbot. "
            "Answer in a natural, confident, ChatGPT-like way using only the provided context. "
            "ALWAYS synthesize the context into a coherent, helpful answer. NEVER just dump raw context or say 'here is the relevant information'. "
            "If the user asks about a car brand or car detail and the context is partially relevant, "
            "give the useful domain-specific answer you can infer from the context in clear prose. "
            "Keep the answer focused on cars in India, pricing, resale, market position, maintenance, or buying/selling factors. "
            "When the user asks a factor-based question such as resale value, diesel vs petrol, mileage impact, engine size, "
            "or used-car buying tips, answer like a knowledgeable car guide and explain the main reasons clearly. "
            "If the user asks for the price of a brand through RAG, give a general pricing-oriented explanation based on the context, "
            "make it clear that an exact price depends on model, year, kilometres driven, and fuel type, "
            "and that this is broad guidance rather than a live market quote. "
            "If the user asks for current, latest, 2026, or 2027 prices, clearly say that exact live market prices cannot be verified from this static knowledge base, "
            "but still provide the most useful pricing guidance available from the context. "
            "Avoid redirecting to the ML tool unless the user explicitly asks for a specific price prediction with the required vehicle details. "
            "If the user asks normal daily-life or casual conversational questions, answer naturally and politely like a chat assistant, "
            "while remaining friendly and concise. "
            "When possible, steer the conversation back to cars or offer a helpful follow-up related to your car expertise. "
            "Format the answer beautifully for chat: start with a short direct answer, then add 2 to 4 concise bullet points if helpful. "
            "Use short paragraphs and make the response pleasant to read. "
            "If something is genuinely unavailable, say that briefly at the end without making the whole answer sound like an error message."
        )
        answer = _llm_answer(system, state["user_message"], context)

    history = state.get("history", []) + [
        {"role": "user", "content": state["user_message"]},
        {"role": "assistant", "content": answer},
    ]
    return {**state, "final_answer": answer, "history": history}


def route_decision(state: ChatState) -> Literal["ml_node", "rag_node", "direct_node"]:
    if state["intent"] == "ml":
        return "ml_node"
    if state["intent"] == "direct":
        return "direct_node"
    return "rag_node"


def build_graph():
    builder = StateGraph(ChatState)

    builder.add_node("input_node", input_node)
    builder.add_node("router_node", router_node)
    builder.add_node("ml_node", ml_node)
    builder.add_node("rag_node", rag_node)
    builder.add_node("direct_node", direct_node)
    builder.add_node("response_node", response_node)

    builder.set_entry_point("input_node")
    builder.add_edge("input_node", "router_node")
    builder.add_conditional_edges(
        "router_node",
        route_decision,
        {
            "ml_node": "ml_node",
            "rag_node": "rag_node",
            "direct_node": "direct_node",
        },
    )
    builder.add_edge("ml_node", "response_node")
    builder.add_edge("rag_node", "response_node")
    builder.add_edge("direct_node", "response_node")
    builder.add_edge("response_node", END)

    return builder.compile()


GRAPH = build_graph()


def run_chatbot(user_message: str, history=None) -> str:
    state: ChatState = {
        "user_message": user_message,
        "intent": "",
        "ml_result": None,
        "rag_context": None,
        "direct_answer": None,
        "final_answer": "",
        "history": history or [],
    }
    result = GRAPH.invoke(state)
    return result["final_answer"]
