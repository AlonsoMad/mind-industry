import pandas as pd
import os

# Define the dataset
data = [
    # 1. AI (Matching)
    {"id_preproc": "Tech_EN_ES_0", "lang": "EN", "title": "Generative AI Progress", "text": "Generative AI models have seen massive improvements in the last few years, especially with transformers."},
    {"id_preproc": "Tech_EN_ES_1", "lang": "ES", "title": "Progreso de la IA Generativa", "text": "Los modelos de IA generativa han visto mejoras masivas en los últimos años, especialmente con los transformadores."},
    
    # 2. Quantum Computing (Matching)
    {"id_preproc": "Tech_EN_ES_2", "lang": "EN", "title": "Quantum Supremacy", "text": "Quantum computers leverage principles of quantum mechanics to process information fundamentally differently than classical computers."},
    {"id_preproc": "Tech_EN_ES_3", "lang": "ES", "title": "Supremacía Cuántica", "text": "Los ordenadores cuánticos aprovechan los principios de la mecánica cuántica para procesar información de forma fundamentalmente distinta a los ordenadores clásicos."},
    
    # 3. Blockchain (Contradiction 1)
    {"id_preproc": "Tech_EN_ES_4", "lang": "EN", "title": "Bitcoin Consensus", "text": "Bitcoin currently uses a Proof of Work consensus mechanism which is highly energy-intensive to secure the network."},
    {"id_preproc": "Tech_EN_ES_5", "lang": "ES", "title": "Consenso de Bitcoin", "text": "Bitcoin utiliza actualmente un mecanismo de consenso de Prueba de Participación (Proof of Stake), el cual es muy eficiente energéticamente para asegurar la red."},
    
    # 4. Web Dev (Matching)
    {"id_preproc": "Tech_EN_ES_6", "lang": "EN", "title": "Frontend Frameworks", "text": "React and Vue.js continue to dominate the frontend development landscape for single page applications."},
    {"id_preproc": "Tech_EN_ES_7", "lang": "ES", "title": "Frameworks de Frontend", "text": "React y Vue.js siguen dominando el panorama del desarrollo frontend para aplicaciones de una sola página."},
    
    # 5. Cyber Security (Contradiction 2)
    {"id_preproc": "Tech_EN_ES_8", "lang": "EN", "title": "Company Firewall Status", "text": "The main corporate firewall rules were successfully updated yesterday to patch all known vulnerabilities."},
    {"id_preproc": "Tech_EN_ES_9", "lang": "ES", "title": "Estado del Cortafuegos de la Empresa", "text": "Las reglas principales del cortafuegos corporativo no han sido actualizadas en el último año, dejando los sistemas vulnerables."},
    
    # 6. Cloud Computing (Matching)
    {"id_preproc": "Tech_EN_ES_10", "lang": "EN", "title": "Serverless Architectures", "text": "Serverless architectures allow developers to build and run applications without thinking about servers."},
    {"id_preproc": "Tech_EN_ES_11", "lang": "ES", "title": "Arquitecturas Serverless", "text": "Las arquitecturas serverless permiten a los desarrolladores crear y ejecutar aplicaciones sin preocuparse por los servidores."},
    
    # 7. 5G Networks (Matching)
    {"id_preproc": "Tech_EN_ES_12", "lang": "EN", "title": "5G Expansion", "text": "The rapid global rollout of 5G networks confidently promises lower latency and higher bandwidth for users."},
    {"id_preproc": "Tech_EN_ES_13", "lang": "ES", "title": "Expansión 5G", "text": "El despliegue de las redes 5G promete menor latencia y mayor ancho de banda para los usuarios móviles a nivel global."},
    
    # 8. Autonomous Vehicles (Contradiction 3)
    {"id_preproc": "Tech_EN_ES_14", "lang": "EN", "title": "Tesla Autopilot Sensors", "text": "Tesla's autonomous driving approach relies entirely on purely vision-based systems with high-resolution cameras, having abandoned radar and LiDAR entirely."},
    {"id_preproc": "Tech_EN_ES_15", "lang": "ES", "title": "Sensores del Autopilot de Tesla", "text": "El enfoque de conducción autónoma de Tesla depende fuertemente del uso de sensores LiDAR para mapear el entorno y asegurar una navegación segura."},
    
    # 9. Semiconductors (Contradiction 4)
    {"id_preproc": "Tech_EN_ES_16", "lang": "EN", "title": "Future of Moore's Law", "text": "Industry experts agree that Moore's Law is effectively dead due to the absolute physical limits of silicon atom size."},
    {"id_preproc": "Tech_EN_ES_17", "lang": "ES", "title": "El futuro de la Ley de Moore", "text": "Los expertos de la industria coinciden en que la Ley de Moore sigue vigente y se está acelerando gracias a los nuevos avances en el empaquetado de chips 3D."},
    
    # 10. Space Tech (Contradiction 5)
    {"id_preproc": "Tech_EN_ES_18", "lang": "EN", "title": "Starship Maiden Flight", "text": "SpaceX's Starship successfully reached orbit, completing all stated objectives on its highly anticipated maiden flight without any issues."},
    {"id_preproc": "Tech_EN_ES_19", "lang": "ES", "title": "Vuelo Inaugural de Starship", "text": "La nave Starship de SpaceX explotó poco después del despegue en su esperado primer vuelo, fallando en alcanzar la órbita terrestre."},

    # 11. TEMPORAL_DISCREPANCY — Product Launch Date
    {"id_preproc": "Tech_EN_ES_20", "lang": "EN", "title": "GPT-5 Release Timeline", "text": "OpenAI officially released GPT-5 in March 2025, making it available to all ChatGPT Plus subscribers on the first day."},
    {"id_preproc": "Tech_EN_ES_21", "lang": "ES", "title": "Fecha de Lanzamiento de GPT-5", "text": "OpenAI lanzó oficialmente GPT-5 en septiembre de 2025, tras varios meses de retraso respecto a la fecha inicialmente prevista."},

    # 12. TEMPORAL_DISCREPANCY — Protocol Adoption
    {"id_preproc": "Tech_EN_ES_22", "lang": "EN", "title": "HTTP/3 Industry Adoption", "text": "The HTTP/3 protocol was widely adopted by major cloud providers starting in 2022, with Google and Cloudflare leading the rollout."},
    {"id_preproc": "Tech_EN_ES_23", "lang": "ES", "title": "Adopción de HTTP/3 en la Industria", "text": "El protocolo HTTP/3 no comenzó a ser adoptado por los principales proveedores de nube hasta finales de 2024, siendo Cloudflare el primero en implementarlo de forma general."},

    # 13. TEMPORAL_DISCREPANCY — Regulation Enactment
    {"id_preproc": "Tech_EN_ES_24", "lang": "EN", "title": "EU AI Act Enforcement", "text": "The European Union's AI Act entered into full legal force in January 2025, requiring all high-risk AI systems to comply immediately."},
    {"id_preproc": "Tech_EN_ES_25", "lang": "ES", "title": "Entrada en Vigor de la Ley de IA de la UE", "text": "La Ley de Inteligencia Artificial de la Unión Europea entró en vigor de forma escalonada, con los requisitos para sistemas de alto riesgo aplicándose a partir de agosto de 2026."},

    # 14. TEMPORAL_DISCREPANCY — Platform Shutdown
    {"id_preproc": "Tech_EN_ES_26", "lang": "EN", "title": "Google Stadia Closure", "text": "Google shut down its Stadia cloud gaming service in January 2023, fully refunding all hardware and game purchases to users."},
    {"id_preproc": "Tech_EN_ES_27", "lang": "ES", "title": "Cierre de Google Stadia", "text": "Google cerró su servicio de juegos en la nube Stadia en marzo de 2024, ofreciendo a los usuarios créditos parciales en la Google Play Store en lugar de reembolsos completos."},

    # 15. TEMPORAL_DISCREPANCY — Chip Fabrication
    {"id_preproc": "Tech_EN_ES_28", "lang": "EN", "title": "TSMC 2nm Production", "text": "TSMC began mass production of 2-nanometer chips in the second half of 2025, supplying Apple as its first major customer."},
    {"id_preproc": "Tech_EN_ES_29", "lang": "ES", "title": "Producción de 2nm de TSMC", "text": "TSMC no iniciará la producción en masa de chips de 2 nanómetros hasta 2027, según declaraciones de su director ejecutivo realizadas a mediados de 2025."},
]

df = pd.DataFrame(data)

# Reorder columns to match original: id_preproc, text, lang, title
df = df[['id_preproc', 'text', 'lang', 'title']]

print(f"Dataset generated with {len(df)} records.")
print(df.info())

# Ensure output directory exists
out_dir = 'Tech_EN_ES_temp'
os.makedirs(out_dir, exist_ok=True)
df.to_parquet(os.path.join(out_dir, 'dataset'), engine='pyarrow')
print(f"Dataset saved to {out_dir}/dataset")
