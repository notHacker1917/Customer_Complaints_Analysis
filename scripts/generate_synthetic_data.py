"""Generate a realistic synthetic Mercedes-Benz multilingual feedback CSV."""
from __future__ import annotations
import csv, random
from pathlib import Path

random.seed(42)

TEMPLATES = [
    ("interior", "en", "The driver seat heater stops working after {x} minutes of driving."),
    ("interior", "en", "Passenger seat ventilation makes a loud rattling noise on bumpy roads."),
    ("interior", "en", "Ambient lighting flickers randomly at night, very distracting."),
    ("interior", "en", "Sunroof refuses to close in cold weather, error on dashboard."),
    ("interior", "en", "Power window switch on driver door is unresponsive intermittently."),
    ("interior", "en", "Cup holder cracked after {x} months of normal use."),
    ("interior", "en", "Steering wheel heater only warms the lower half of the rim."),
    ("interior", "en", "Rear seat belt buckle is stuck, cannot release the belt."),
    ("interior", "en", "Climate control blows cold air on driver side and hot on passenger side."),
    ("interior", "en", "Dome light stays on after locking the car, drains the 12v battery."),
    ("interior", "en", "Glove box latch broke, lid will not stay closed."),
    ("interior", "en", "Center console armrest is loose and wobbles when leaning on it."),
    ("interior", "en", "Cabin filter smells musty even after replacement."),
    ("interior", "en", "Driver seat cushion has worn through after {x} months."),
    ("interior", "en", "Mirror auto-dimming feature does not work at night."),
    ("powertrain", "en", "Engine misfire under heavy acceleration, check engine light on."),
    ("powertrain", "en", "Turbocharger whines loudly above {x}000 rpm."),
    ("powertrain", "en", "9G transmission shifts hard between 2nd and 3rd gear when cold."),
    ("powertrain", "en", "AdBlue warning appears constantly even after refilling."),
    ("powertrain", "en", "12v battery drains overnight, car will not start in the morning."),
    ("powertrain", "en", "Coolant leak under the engine bay, sweet smell after driving."),
    ("powertrain", "en", "EV high voltage battery range dropped {x} percent after one winter."),
    ("powertrain", "en", "Electric motor produces a humming noise when accelerating from stop."),
    ("powertrain", "en", "On-board charger refuses to connect with public DC fast chargers."),
    ("powertrain", "en", "Diesel particulate filter regeneration fails repeatedly."),
    ("powertrain", "de", "Der Turbolader pfeift laut bei hoher Drehzahl."),
    ("powertrain", "de", "Das 9G-Getriebe schaltet im kalten Zustand sehr hart."),
    ("powertrain", "de", "AdBlue-Warnung bleibt trotz Nachfuellen bestehen."),
    ("powertrain", "de", "Die 12V-Batterie entlaedt sich ueber Nacht komplett."),
    ("powertrain", "de", "Kuehlmittelverlust unter dem Motor, suesslicher Geruch."),
    ("powertrain", "de", "Reichweite der Hochvoltbatterie ist nach einem Winter um {x} Prozent gesunken."),
    ("powertrain", "de", "Der Elektromotor brummt beim Anfahren aus dem Stand."),
    ("powertrain", "de", "Dieselpartikelfilter-Regeneration schlaegt wiederholt fehl."),
    ("display_infotainment", "en", "MBUX touchscreen freezes during navigation, requires reboot."),
    ("display_infotainment", "en", "Apple CarPlay disconnects every {x} minutes when using maps."),
    ("display_infotainment", "en", "Hey Mercedes voice assistant does not recognize commands."),
    ("display_infotainment", "en", "Backup camera shows black screen when reverse is engaged."),
    ("display_infotainment", "en", "Burmester audio system has static noise on right rear speaker."),
    ("display_infotainment", "en", "Wireless charger overheats and stops charging the phone."),
    ("display_infotainment", "en", "Head-up display flickers in direct sunlight."),
    ("display_infotainment", "en", "Software update OTA failed and bricked the head unit."),
    ("display_infotainment", "en", "Bluetooth pairing drops every time the car is restarted."),
    ("display_infotainment", "en", "GPS navigation routes through closed roads repeatedly."),
    ("display_infotainment", "de", "MBUX-Bildschirm friert waehrend der Navigation ein."),
    ("display_infotainment", "de", "Apple CarPlay trennt die Verbindung alle {x} Minuten."),
    ("display_infotainment", "de", "Hey Mercedes Sprachassistent versteht Befehle nicht."),
    ("display_infotainment", "de", "Rueckfahrkamera zeigt schwarzen Bildschirm beim Einlegen des Rueckwaertsgangs."),
    ("display_infotainment", "de", "Burmester Soundsystem hat statisches Rauschen am hinteren rechten Lautsprecher."),
    ("display_infotainment", "de", "Drahtloses Ladegeraet ueberhitzt und stoppt das Laden."),
    ("display_infotainment", "da", "MBUX skaermen fryser under navigation, skal genstartes."),
    ("display_infotainment", "da", "Bluetooth-forbindelsen afbrydes hver gang bilen genstartes."),
    ("display_infotainment", "da", "Bakkamera viser sort skaerm naar jeg saetter i bakgear."),
    ("display_infotainment", "tr", "MBUX dokunmatik ekran navigasyon sirasinda donuyor."),
    ("display_infotainment", "tr", "Apple CarPlay her {x} dakikada bir baglantisi kesiliyor."),
    ("display_infotainment", "tr", "Geri vites kamerasi siyah ekran gosteriyor."),
    ("display_infotainment", "pl", "Ekran MBUX zawiesza sie podczas nawigacji, wymaga restartu."),
    ("display_infotainment", "pl", "Apple CarPlay rozlacza sie co {x} minut."),
    ("display_infotainment", "pl", "Asystent glosowy Hey Mercedes nie rozpoznaje polecen."),
    ("display_infotainment", "fr", "L ecran MBUX se fige pendant la navigation, redemarrage requis."),
    ("display_infotainment", "fr", "La camera de recul affiche un ecran noir."),
    ("display_infotainment", "it", "Lo schermo MBUX si blocca durante la navigazione."),
    ("display_infotainment", "it", "Apple CarPlay si disconnette ogni {x} minuti."),
    ("display_infotainment", "es", "La pantalla MBUX se congela durante la navegacion."),
    ("display_infotainment", "es", "La camara de marcha atras muestra pantalla negra."),
]

NOISE_SUFFIXES = ["", "", "", "", "",
                  " Please fix this asap.",
                  " Mercedes please help.",
                  " Contact me at john.doe@example.com or +49 30 1234567.",
                  " VIN: WDB2030461A123456",
                  " Plate: B-MW 1234",
                  " !!!!!!",
                  " <p>this is a complaint</p>",
                  "  multiple   spaces   here  "]

def fill(t):
    return t.format(x=random.randint(2,60)) if "{x}" in t else t

def main(out_path="data/raw/all_feedback.csv", n_total=2500):
    out = Path(out_path); out.parent.mkdir(parents=True, exist_ok=True)
    by_domain = {"interior": [], "powertrain": [], "display_infotainment": []}
    for t in TEMPLATES: by_domain[t[0]].append(t)
    weights = {"interior":0.70, "powertrain":0.08, "display_infotainment":0.22}
    rows = []
    for i in range(n_total):
        d = random.choices(list(weights.keys()), weights=list(weights.values()))[0]
        _d, lang, tmpl = random.choice(by_domain[d])
        text = fill(tmpl) + random.choice(NOISE_SUFFIXES)
        rows.append({"record_id": f"R{i:06d}", "domain": d,
                     "language_hint": lang, "feedback": text})
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["record_id","domain","language_hint","feedback"])
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {len(rows):,} rows -> {out}")

if __name__ == "__main__":
    main()
