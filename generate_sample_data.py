"""
generate_sample_data.py
=======================
Create realistic synthetic Mercedes-Benz customer feedback CSVs that
mimic the three production domains.

Test sizes are small so the pipeline finishes in seconds on CPU. To
stress-test, bump SIZES to the spec's {75000, 5000, 29000}.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

random.seed(7)

OUT_DIR = Path(__file__).parent / "data" / "raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SIZES = {"interior": 700, "powertrain": 250, "display_infotainment": 550}

INTERIOR_EN = [
    "The driver seat heater stopped working after {months} months, only the passenger side gets warm.",
    "Power window switch on the {door} door is stuck and the window won't go down.",
    "Seat belt on the rear left is jammed and won't retract properly.",
    "Sunroof makes a loud rattling noise when driving over bumps.",
    "Ambient lighting flickers randomly at night, mostly around the dashboard area.",
    "Air conditioning blows warm air on the driver side even when set to cold.",
    "The leather on the steering wheel is peeling after only {months} months.",
    "Cup holder in the center console is cracked and no longer holds bottles securely.",
    "Glove box latch is broken, the lid keeps falling open while driving.",
    "Headrest in the passenger seat will not adjust up or down.",
    "Cabin filter smells musty even after replacement.",
    "Floor mat on the driver side keeps sliding forward and blocking the pedals.",
    "Trunk latch fails to release using the key fob, only manual opening works.",
    "Rear seat heater is unresponsive on the left side, right side works fine.",
    "Dashboard creaks loudly especially in cold weather.",
    "Side mirror does not fold automatically when locking the car.",
    "Reading light over the rear seat does not turn on at all.",
    "Sun visor on the driver side hangs loose and will not stay up.",
    "Seat ventilation makes a high-pitched whining noise.",
    "Door lock on the rear right door clicks repeatedly when locking.",
]

POWERTRAIN_EN = [
    "Engine stalls at idle when the AC is on, especially in traffic.",
    "Check engine light comes on intermittently with code P{code}.",
    "Transmission shifts hard between 2nd and 3rd gear when cold.",
    "Turbocharger whistling noise getting louder over the past {weeks} weeks.",
    "Battery dies overnight, had to jump start three times this week.",
    "Coolant leak under the engine bay, losing about a liter per week.",
    "AdBlue warning keeps appearing even after a full refill.",
    "Diesel particulate filter regeneration fails, warning message persists.",
    "Electric motor produces a grinding noise during acceleration.",
    "EV battery range dropped {pct} percent in the last month with no driving change.",
    "Charging port cover will not close properly after fast charging.",
    "Spark plug misfire on cylinder {cyl}, engine runs rough.",
    "Oil leak from the valve cover gasket dripping onto the exhaust manifold.",
    "Alternator failed at {km}k km, dashboard lights all flickered before stalling.",
    "Fuel injector clicking sound from cylinder {cyl}, hesitation under load.",
]

POWERTRAIN_DE = [
    "Motor geht im Leerlauf aus, besonders wenn die Klimaanlage laeuft.",
    "Motorkontrollleuchte geht sporadisch an mit Fehlercode P{code}.",
    "Getriebe schaltet hart zwischen 2. und 3. Gang im kalten Zustand.",
    "Turbolader pfeift seit {weeks} Wochen immer lauter.",
    "Batterie ist ueber Nacht leer, musste diese Woche dreimal ueberbruecken.",
    "Kuehlmittelleck unter dem Motorraum, ungefaehr ein Liter pro Woche Verlust.",
    "AdBlue-Warnung bleibt auch nach vollstaendiger Befuellung bestehen.",
    "Dieselpartikelfilter-Regeneration schlaegt fehl, Warnmeldung verschwindet nicht.",
    "Elektromotor macht ein kratzendes Geraeusch beim Beschleunigen.",
    "Reichweite der EV-Batterie um {pct} Prozent gesunken im letzten Monat ohne Fahraenderung.",
]

DISPLAY_EN = [
    "MBUX touchscreen freezes during navigation, requires full vehicle restart.",
    "Central display goes black for a few seconds when starting the car.",
    "Apple CarPlay disconnects every {min} minutes, very frustrating on long drives.",
    "Voice assistant Hey Mercedes does not respond to commands in noisy environments.",
    "Backup camera image is distorted and shows green static lines.",
    "Bluetooth audio cuts out when receiving phone calls.",
    "Wireless phone charger overheats and stops charging after {min} minutes.",
    "Head-up display flickers in direct sunlight, hard to read speed.",
    "Software update failed twice, system stuck on installation screen.",
    "Burmester audio system has crackling noise from rear left speaker.",
    "Navigation directions are delayed by 2-3 seconds, often miss turns.",
    "Instrument cluster shows wrong fuel level, jumps between half and full.",
    "Touchscreen unresponsive in cold weather until interior warms up.",
    "Android Auto fails to launch, screen shows USB icon then disappears.",
    "Telematics module loses cellular connection in tunnels and takes long to recover.",
]

DISPLAY_DE = [
    "MBUX-Touchscreen friert waehrend der Navigation ein, vollstaendiger Neustart noetig.",
    "Zentrales Display wird beim Starten fuer einige Sekunden schwarz.",
    "Apple CarPlay trennt sich alle {min} Minuten, sehr aergerlich auf langen Fahrten.",
    "Sprachassistent Hey Mercedes reagiert in lauter Umgebung nicht auf Befehle.",
    "Rueckfahrkamerabild ist verzerrt mit gruenen statischen Linien.",
    "Bluetooth-Audio bricht ab wenn Anrufe eingehen.",
    "Kabelloses Ladegeraet ueberhitzt und stoppt nach {min} Minuten.",
    "Head-up-Display flackert bei direktem Sonnenlicht.",
]

DISPLAY_DA = [
    "MBUX touchskaerm fryser under navigation, kraever genstart af bilen.",
    "Apple CarPlay afbrydes hvert {min}. minut, meget irriterende.",
    "Bagudkamera viser forvraenget billede med groenne striber.",
    "Tradloes oplader bliver overophedet og stopper efter {min} minutter.",
    "Bluetooth-lyd afbrydes naar jeg modtager telefonopkald.",
]

DISPLAY_TR = [
    "MBUX dokunmatik ekran navigasyon sirasinda donuyor, araci yeniden baslatmak gerekiyor.",
    "Apple CarPlay her {min} dakikada bir baglantisi kesiliyor.",
    "Geri gorus kamerasi bozuk goruntu ve yesil cizgiler gosteriyor.",
    "Kablosuz sarj cihazi isiniyor ve {min} dakika sonra duruyor.",
    "Bluetooth ses telefon aramasi geldiginde kesiliyor.",
]

DISPLAY_PL = [
    "Ekran dotykowy MBUX zawiesza sie podczas nawigacji, wymaga restartu pojazdu.",
    "Apple CarPlay rozlacza sie co {min} minut, bardzo frustrujace.",
    "Kamera cofania pokazuje znieksztalcony obraz z zielonymi liniami.",
    "Ladowarka bezprzewodowa przegrzewa sie i zatrzymuje po {min} minutach.",
    "Dzwiek Bluetooth zanika podczas polaczen telefonicznych.",
]

DISPLAY_FR = [
    "L'ecran tactile MBUX se fige pendant la navigation, redemarrage du vehicule necessaire.",
    "Apple CarPlay se deconnecte toutes les {min} minutes.",
    "La camera de recul affiche une image deformee avec des lignes vertes.",
]

DISPLAY_IT = [
    "Lo schermo touch MBUX si blocca durante la navigazione, richiede il riavvio del veicolo.",
    "Apple CarPlay si disconnette ogni {min} minuti.",
    "La telecamera posteriore mostra un'immagine distorta con linee verdi.",
]

def fill(template: str) -> str:
    return template.format(
        months=random.randint(2, 36),
        weeks=random.randint(2, 12),
        door=random.choice(["front-left", "front-right", "rear-left", "rear-right"]),
        code=random.choice(["0420", "0171", "0300", "0455", "0128"]),
        pct=random.randint(10, 35),
        cyl=random.randint(1, 8),
        km=random.randint(20, 180),
        min=random.choice([5, 10, 15, 20, 30]),
    )

def maybe_noise(s: str) -> str:
    r = random.random()
    if r < 0.05:   s = s.upper()
    elif r < 0.10: s = s + "  !!!  please fix this asap"
    elif r < 0.15: s = "<p>" + s + "</p>"
    elif r < 0.18: s = s + " more info: https://forum.example.com/thread/" + str(random.randint(1000,9999))
    elif r < 0.20: s = s + " contact me at owner" + str(random.randint(1,999)) + "@example.com"
    elif r < 0.22: s = s + " VIN: WDB" + "".join(random.choices("0123456789ABCDEFGH", k=14))
    return s

def gen(n: int, pools: list[list[str]]) -> list[str]:
    return [maybe_noise(fill(random.choice(random.choice(pools)))) for _ in range(n)]

def write_csv(path: Path, rows: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["feedback"])
        for r in rows: w.writerow([r])
    print(f"wrote {len(rows):>6,} rows -> {path}")

def main() -> None:
    write_csv(OUT_DIR / "interior_feedback.csv",   gen(SIZES["interior"], [INTERIOR_EN]))
    write_csv(OUT_DIR / "powertrain_feedback.csv", gen(SIZES["powertrain"], [POWERTRAIN_EN, POWERTRAIN_DE]))
    write_csv(OUT_DIR / "display_feedback.csv",    gen(SIZES["display_infotainment"],
                  [DISPLAY_EN, DISPLAY_DE, DISPLAY_DA, DISPLAY_TR, DISPLAY_PL, DISPLAY_FR, DISPLAY_IT]))

if __name__ == "__main__":
    main()
