"""
Generate a dataset of 200 informal, emotion-rich Mercedes-Benz customer complaints.

Output schema (9 columns):
  complaint_id | vehicle_model | manufacturing_year | mileage_km |
  component    | complaint_text | service_center  | date_reported | detected_language
"""
from __future__ import annotations
import csv, random
from datetime import date, timedelta
from pathlib import Path

random.seed(2026)

# --------------------------------------------------------------- vehicles
MODELS = [
    "A 180", "A 200", "A 250", "A 35 AMG",
    "C 200", "C 220d", "C 300", "C 43 AMG",
    "E 200", "E 220d", "E 300", "E 350", "E 450", "E 53 AMG",
    "S 350d", "S 400d", "S 500", "S 580",
    "GLA 200", "GLB 220", "GLC 300", "GLC 43 AMG", "GLE 350d", "GLE 450",
    "GLS 450", "G 400d", "G 63 AMG",
    "CLA 200", "CLS 350d",
    "EQA 250", "EQB 300", "EQC 400", "EQE 350", "EQS 450", "EQS 580",
    "AMG GT 53", "V 300d",
]

SERVICE_CENTERS = [
    ("Mercedes-Benz Berlin Tempelhof", "DE"),
    ("Stern Auto Hamburg",             "DE"),
    ("Daimler München Mitte",          "DE"),
    ("Mercedes-Benz Stuttgart",        "DE"),
    ("Auto-Heinrich Frankfurt",        "DE"),
    ("Mercedes-Benz Köln Rhein",       "DE"),
    ("Mercedes-Benz of Mayfair London","UK"),
    ("Lookers Mercedes Birmingham",    "UK"),
    ("Sytner Manchester",              "UK"),
    ("Mercedes-Benz Paris Étoile",     "FR"),
    ("Étoile Lyon",                    "FR"),
    ("Mercedes Roma EUR",              "IT"),
    ("Stern Milano Centro",            "IT"),
    ("Autotorino Verona",              "IT"),
    ("Mercedes-Benz Madrid Norte",     "ES"),
    ("Itra Barcelona",                 "ES"),
    ("Mercedes Amsterdam Zuid",        "NL"),
    ("Mercedes-Benz Copenhagen",       "DK"),
    ("Koluman Istanbul Ataşehir",      "TR"),
    ("Mercedes-Benz Ankara Çankaya",   "TR"),
    ("Auto Wimar Warsaw",              "PL"),
    ("Mercedes-Benz Poznań",           "PL"),
    ("Mercedes-Benz Brussels",         "BE"),
    ("Pappas Wien",                    "AT"),
    ("Merbag Zürich",                  "CH"),
]

COMPONENTS = [
    "MBUX touchscreen", "Apple CarPlay", "Bluetooth module", "Hey Mercedes assistant",
    "backup camera", "head-up display", "Burmester audio", "wireless charger",
    "navigation system", "infotainment software", "instrument cluster",
    "driver seat heater", "passenger seat heater", "seat ventilation", "seat cushion",
    "steering wheel heater", "ambient lighting", "sunroof", "panoramic roof",
    "climate control", "cabin filter", "dome light", "power window switch",
    "cup holder", "glove box", "armrest", "auto-dimming mirror",
    "engine", "turbocharger", "9G transmission", "12V battery", "high voltage battery",
    "AdBlue system", "DPF filter", "electric motor", "on-board charger",
    "coolant system", "fuel injector",
]

# ----------------------------------------------------- emotional templates
# Each tuple: (language_code, complaint_text)
# Templates cover: frustration, anger, sarcasm, worry, resignation, pleading.
TEMPLATES = [
    # ============ ENGLISH (~50%) ============
    ("en", "Honestly?? Bought my {model} last year and the MBUX screen is already glitching. Disappointed doesn't even cover it."),
    ("en", "I'm SO over this. Driver seat heater stopped working AGAIN, third time in 6 months. What am I even paying for??"),
    ("en", "Look, I love my Mercedes but this turbo whine is driving me crazy. Please tell me this isn't normal."),
    ("en", "Apple CarPlay disconnects every. single. time. I get on the highway. WHY does this keep happening"),
    ("en", "Stuck on the side of the road because my 12V battery just... died. In a 2024 {model}. Make it make sense."),
    ("en", "Cup holder cracked. Yeah, the cup holder. €60k car and the cup holder cracked. Living the dream."),
    ("en", "My {model} smells like a wet dog after every rain. Whatever it is in the cabin filter, fix it please I'm begging."),
    ("en", "Okay so the sunroof refuses to close in cold weather. So if I park outside in winter... fun times!"),
    ("en", "Thought I was crazy but 3 friends with the same model said the same thing. MBUX freezes during navigation. You guys know about this right??"),
    ("en", "Genuinely scared to drive on the autobahn now. The 9G transmission shifts so hard sometimes it feels like I got rear-ended."),
    ("en", "Please please please tell me what to do. Hey Mercedes literally never understands a single word I say. I just feel stupid talking to my car."),
    ("en", "Frustrated beyond words. Backup camera shows a black screen every time it rains. So basically useless when I actually need it."),
    ("en", "My wife is afraid to ride with me anymore. The engine misfires randomly and the check engine light has been on since week 2."),
    ("en", "I just want to vent — climate control blows ice cold on driver side and SAUNA on passenger side. Three visits, no fix. I'm done."),
    ("en", "Bro the AdBlue warning light has been on for SIX months even though I refilled it 4 times. What is this nonsense"),
    ("en", "Burmester audio system has static on the right rear speaker. €4000 option btw. Cool cool cool."),
    ("en", "Dome light won't turn off after locking the car, drained the battery twice. I now sleep with one eye on my driveway."),
    ("en", "MBUX bricked itself after the OTA update last week. Yes, BRICKED. Beautiful 12 inch black screen now."),
    ("en", "Steering wheel heater warms only the bottom half of the rim. So my palms freeze. Engineering at its finest!"),
    ("en", "Coolant leaking under the engine bay, sweet smell, pretty sure it's the radiator. 38k km on a 2023 car. Not impressed."),
    ("en", "Ambient lighting flickers like a cheap nightclub. At night. While driving. Disorienting and honestly dangerous."),
    ("en", "I literally cried when the EV battery range dropped 28% over one winter. I traded in my BMW for THIS?"),
    ("en", "Mirror auto-dim doesn't work. Headlights from the car behind blind me every night. I dread night driving now."),
    ("en", "Glove box latch broke. Sounds minor — until your insurance papers are flying around the cabin every time you brake."),
    ("en", "The wireless charger overheats my phone to the point I can't touch it. Genuinely worried it'll start a fire."),
    ("en", "9G gearbox slips going into 3rd. Service says 'within tolerance'. WHOSE tolerance, exactly??"),
    ("en", "DPF regen fails repeatedly, warning light, then service mode. Lost a whole afternoon of work because of this."),
    ("en", "Door panel rattles louder than the music. €70k car, sounds like a Lada from 1989."),
    ("en", "Power window on driver side decides whether to work each morning. It's like a coin flip. A €600 coin flip when it dies fully."),
    ("en", "Bluetooth pairing drops EVERY restart. So every morning I get to manually re-pair my phone. Riveting."),
    ("en", "Heads-up display flickers in direct sunlight which, if I may, IS WHEN I MOST NEED IT."),
    ("en", "Cabin smells like burning plastic on cold starts. Three trips to dealer, three 'no fault found'. Yeah okay."),

    # ============ GERMAN (~20%) ============
    ("de", "Ehrlich gesagt nur noch genervt. Der Turbolader pfeift wie verrückt seit 30.000 km. Hilfe?"),
    ("de", "Ich liebe meinen Benz aber das ist langsam echt eine Frechheit. Sitzheizung geht alle 2 Wochen kaputt."),
    ("de", "Komme gerade aus der Werkstatt. ZUM DRITTEN MAL wegen AdBlue. Das geht mir mittlerweile richtig auf den Geist."),
    ("de", "Der Bildschirm friert mitten in der Navi ein und ich soll mal eben anhalten?? Mercedes bitte, das ist 2026."),
    ("de", "Mein {model} ist Baujahr 2024 und die 12V-Batterie war schon zwei Mal leer. Was soll das?"),
    ("de", "Ich weine fast wenn ich an die Reichweite denke. Im Winter -32%. Hätte einen Tesla nehmen sollen ehrlich."),
    ("de", "Hey Mercedes versteht KEIN Wort von mir. Mein Hund hört besser zu. Echt jetzt."),
    ("de", "Klimaanlage bläst kalt links, warm rechts. Drei Werkstattbesuche, kein Erfolg. Ich resigniere."),
    ("de", "Der Kühlmittelverlust macht mich wahnsinnig. Süßer Geruch, Pfütze unter dem Auto, und keiner findet was."),
    ("de", "Apple CarPlay trennt sich alle 5 Minuten. Das ist nicht mehr lustig, das ist ein Witz."),
    ("de", "Ambiente-Beleuchtung flackert wie eine kaputte Discokugel. Schämen Sie sich nicht?"),
    ("de", "Der Becherhalter ist gebrochen. BECHERHALTER. In einem 80.000 € Auto. Ich kann nicht mehr."),
    ("de", "Lenkradheizung wärmt nur unten. Meine Daumen frieren weiterhin tapfer mit."),
    ("de", "MBUX-Update hat das Kombiinstrument zerschossen. Ich fahre jetzt ohne Tacho. Toll."),

    # ============ DANISH ============
    ("da", "Helt ærligt jeg er træt af det her. MBUX skærmen fryser hver gang jeg bruger Apple CarPlay. Hvad sker der?"),
    ("da", "Min splinternye {model} og bakkameraet virker ikke i regnvejr. SERIØST?"),
    ("da", "Sædevarmen fungerer kun nogle gange. Tilfældigt. Jeg har givet op og købt et tæppe."),
    ("da", "12V batteriet er dødt for tredje gang. På to år. Jeg er færdig med Mercedes."),
    ("da", "Hey Mercedes forstår ikke et ord. Hvorfor sælger I det her som en feature?"),

    # ============ TURKISH ============
    ("tr", "Yav arkadaşlar bu ne ya, MBUX ekranı sürekli donuyor. 2 milyonluk araba bu!"),
    ("tr", "Apple CarPlay her 5 dakikada bağlantıyı kesiyor, çıldıracağım resmen!"),
    ("tr", "Direksiyon ısıtması çalışmıyor, servis 'normal' diyor. Hangi normalde?"),
    ("tr", "Geri vites kamerası karanlıkta hiçbir şey göstermiyor. Tehlikeli bence."),
    ("tr", "Yağ kokusu motordan geliyor, üç kez serviste hala çözüm yok. Pes."),
    ("tr", "Hey Mercedes Türkçe anlamıyor zaten ama İngilizce de anlamıyor. Komik."),

    # ============ POLISH ============
    ("pl", "No serio?? Kupiłem nowego Mercedesa i bluetooth się rozłącza co minutę. Słabo to wygląda."),
    ("pl", "Klimatyzacja dmucha zimnym z jednej strony, gorącym z drugiej. To jest jakiś żart??"),
    ("pl", "Akumulator 12V padł trzeci raz. Trzeci. W aucie z 2024 roku. Niesamowite."),
    ("pl", "MBUX zawiesza się podczas nawigacji, muszę zatrzymywać auto żeby zrestartować. Genialne."),
    ("pl", "Kamera cofania pokazuje czarny ekran w deszczu. Czyli wtedy gdy najbardziej potrzebna."),

    # ============ FRENCH ============
    ("fr", "Sérieusement?? L'écran MBUX plante sans arrêt et le concessionnaire dit que c'est normal. NON c'est PAS normal."),
    ("fr", "Mon turbo siffle comme un train, j'en peux plus. Trois passages au garage, zéro résultat."),
    ("fr", "Le siège chauffant ne marche que d'un côté. Mes fesses sont confuses."),
    ("fr", "Apple CarPlay se déconnecte toutes les 3 minutes. C'est devenu un sport quotidien."),
    ("fr", "La caméra de recul est noire dès qu'il pleut. Donc inutile 6 mois par an en France. Bravo."),

    # ============ ITALIAN ============
    ("it", "Allora ragazzi, la mia {model} gira solo aria fredda dal lato passeggero. Pagato 70k per questo??"),
    ("it", "Apple CarPlay si scollega ogni 2 minuti, sto impazzendo davvero."),
    ("it", "Lo schermo MBUX si blocca durante la navigazione e devo fermarmi a riavviare. Vergogna."),
    ("it", "Il filtro dell'aria puzza di muffa anche dopo la sostituzione. Ma scherziamo?"),
    ("it", "Tre visite per la stessa perdita di liquido refrigerante. Sempre 'non riscontrato'. Sono sfinito."),

    # ============ SPANISH ============
    ("es", "En serio? Mi {model} nuevo y la pantalla MBUX se congela todos los días. Estoy harto."),
    ("es", "El asistente Hey Mercedes no entiende nada de lo que digo. Cero útil. Cero."),
    ("es", "El turbo silba como una tetera al hervir. Tres talleres, ningún diagnóstico. Resignado."),
    ("es", "La cámara de marcha atrás se ve negra cuando llueve. Justo cuando la necesito. Magnífico."),
    ("es", "La calefacción del asiento solo funciona el lunes (broma). En realidad funciona aleatoriamente."),

    # ============ DUTCH ============
    ("nl", "Eerlijk gezegd ben ik er klaar mee. MBUX bevriest steeds tijdens navigatie."),
    ("nl", "Mijn nieuwe {model} en het bekertje is al gebarsten. Echt een grap."),
    ("nl", "Apple CarPlay verbreekt elke 5 minuten. Dit is geen luxewagen, dit is hoofdpijn."),

    # ============ MIXED / CODE-SWITCH (realistic) ============
    ("de", "Mein Händler sagt 'das ist normal'. Sorry aber NEIN, das ist NICHT normal wenn der Bildschirm 3x am Tag abstürzt."),
    ("en", "Service centre keeps gaslighting me about the rattle. Nope, my ears work fine, the panel IS loose."),
    ("fr", "Mon concessionnaire me prend pour une idiote. 'Madame, c'est normal' — NON, monsieur, ce n'est PAS normal."),
    ("en", "I'm 6 months out of warranty. Failed turbo. €4200 quote. Tell me again why I bought premium?"),
    ("de", "Garantie eine Woche abgelaufen. Getriebe kaputt. Kostenvoranschlag 6.800 €. Ich kotze."),
]


def random_mileage(year: int) -> int:
    today_year = 2026
    age = max(today_year - year, 0)
    avg_per_year = random.choice([8000, 12000, 15000, 18000, 22000, 28000])
    base = age * avg_per_year
    # add jitter
    base = int(base * random.uniform(0.6, 1.4))
    if year == 2026: base = random.randint(50, 4000)
    return max(base, 50)


def random_date() -> str:
    end = date(2026, 5, 9)
    start = date(2024, 11, 1)
    delta_days = (end - start).days
    return (start + timedelta(days=random.randint(0, delta_days))).isoformat()


def main(out_path: str = "data/raw/emotional_complaints_200.csv", n: int = 200):
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(n):
        lang, template = random.choice(TEMPLATES)
        model = random.choice(MODELS)
        text = template.format(model=model)
        # Pick a component that loosely matches the keywords if possible,
        # else pick at random for realistic noise
        comp_match = next((c for c in COMPONENTS if c.lower() in text.lower()), None)
        component = comp_match or random.choice(COMPONENTS)
        year = random.choices(
            [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026],
            weights=[3, 4, 6, 8, 11, 14, 18, 22, 14])[0]
        center, _country = random.choice(SERVICE_CENTERS)
        rows.append({
            "complaint_id":        f"C{i+1:05d}",
            "vehicle_model":       model,
            "manufacturing_year":  year,
            "mileage_km":          random_mileage(year),
            "component":           component,
            "complaint_text":      text,
            "service_center":      center,
            "date_reported":       random_date(),
            "detected_language":   lang,
        })

    fields = ["complaint_id", "vehicle_model", "manufacturing_year", "mileage_km",
              "component", "complaint_text", "service_center", "date_reported",
              "detected_language"]
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
