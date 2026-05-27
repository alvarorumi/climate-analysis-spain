"""
Análisis de Temperaturas en España (1950-2024)
¿Se están calentando nuestras ciudades?
Autor: Álvaro Rubio Milán
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import requests
from datetime import datetime

# ─────────────────────────────────────────────
# 1. CONFIGURACIÓN DE CIUDADES
# ─────────────────────────────────────────────

CITIES = {
    "Madrid":    {"lat": 40.42, "lon": -3.70},
    "Barcelona": {"lat": 41.39, "lon":  2.17},
    "Sevilla":   {"lat": 37.39, "lon": -5.99},
    "Granada":   {"lat": 37.18, "lon": -3.60},
    "Bilbao":    {"lat": 43.26, "lon": -2.93},
}

START_YEAR = 1950
END_YEAR   = 2024

# ─────────────────────────────────────────────
# 2. DESCARGA DE DATOS (Open-Meteo, gratis)
# ─────────────────────────────────────────────

def fetch_temperature_data(city, lat, lon, start_year, end_year):
    """Descarga temperatura media diaria de Open-Meteo para una ciudad."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":        lat,
        "longitude":       lon,
        "start_date":      f"{start_year}-01-01",
        "end_date":        f"{end_year}-12-31",
        "daily":           "temperature_2m_mean",
        "timezone":        "Europe/Madrid",
    }
    print(f"  Descargando datos de {city}...")
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame({
        "date": pd.to_datetime(data["daily"]["time"]),
        "temp": data["daily"]["temperature_2m_mean"],
    })
    df["city"]  = city
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month
    return df


print("=" * 55)
print("  ANÁLISIS DE TEMPERATURAS EN ESPAÑA (1950-2024)")
print("=" * 55)
print("\n📡 Descargando datos históricos...\n")

frames = []
for city, coords in CITIES.items():
    df = fetch_temperature_data(city, coords["lat"], coords["lon"], START_YEAR, END_YEAR)
    frames.append(df)

raw = pd.concat(frames, ignore_index=True)
raw.dropna(subset=["temp"], inplace=True)

print(f"\n✅ Datos descargados: {len(raw):,} registros diarios\n")

# ─────────────────────────────────────────────
# 3. ANÁLISIS ESTADÍSTICO
# ─────────────────────────────────────────────

# Temperatura media anual por ciudad
annual = (
    raw.groupby(["city", "year"])["temp"]
    .mean()
    .reset_index()
    .rename(columns={"temp": "annual_mean"})
)

# Temperatura media por década
def decade(year):
    return f"{(year // 10) * 10}s"

annual["decade"] = annual["year"].apply(decade)

decade_stats = (
    annual.groupby(["city", "decade"])["annual_mean"]
    .mean()
    .reset_index()
)

# Diferencia entre primera y última década
first_decade = str((START_YEAR // 10) * 10) + "s"
last_decade  = str((END_YEAR   // 10) * 10) + "s"

warming = {}
for city in CITIES:
    t0 = decade_stats.query("city == @city and decade == @first_decade")["annual_mean"].values
    t1 = decade_stats.query("city == @city and decade == @last_decade")["annual_mean"].values
    if len(t0) and len(t1):
        warming[city] = round(t1[0] - t0[0], 2)

print("🌡️  Calentamiento estimado (1950s vs 2020s):")
for city, delta in sorted(warming.items(), key=lambda x: -x[1]):
    bar = "█" * int(abs(delta) * 4)
    print(f"   {city:<12} +{delta}°C  {bar}")

# Tendencia lineal (regresión simple)
from numpy.polynomial import polynomial as P
import numpy as np

trends = {}
for city in CITIES:
    sub = annual[annual["city"] == city].dropna()
    coeffs = np.polyfit(sub["year"], sub["annual_mean"], 1)
    trends[city] = round(coeffs[0] * 10, 3)   # °C por década

print("\n📈 Tendencia de calentamiento (°C por década):")
for city, trend in sorted(trends.items(), key=lambda x: -x[1]):
    print(f"   {city:<12} +{trend}°C/década")

# ─────────────────────────────────────────────
# 4. VISUALIZACIONES
# ─────────────────────────────────────────────

palette = {
    "Madrid":    "#E74C3C",
    "Barcelona": "#3498DB",
    "Sevilla":   "#F39C12",
    "Granada":   "#8E44AD",
    "Bilbao":    "#27AE60",
}

plt.style.use("seaborn-v0_8-whitegrid")
fig = plt.figure(figsize=(18, 14))
fig.suptitle(
    "¿Se están calentando las ciudades españolas? (1950–2024)",
    fontsize=18, fontweight="bold", y=0.98
)
gs = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.32)

# ── Gráfico 1: Temperatura media anual con tendencia ──
ax1 = fig.add_subplot(gs[0, :])
for city in CITIES:
    sub = annual[annual["city"] == city]
    # Suavizado 10 años
    sub = sub.sort_values("year").copy()
    sub["smooth"] = sub["annual_mean"].rolling(10, center=True, min_periods=5).mean()
    ax1.plot(sub["year"], sub["annual_mean"], alpha=0.18, color=palette[city], linewidth=1)
    ax1.plot(sub["year"], sub["smooth"], color=palette[city], linewidth=2.2, label=city)

ax1.set_title("Temperatura media anual (media móvil 10 años)", fontsize=13, pad=10)
ax1.set_xlabel("Año")
ax1.set_ylabel("Temperatura (°C)")
ax1.legend(loc="upper left", framealpha=0.9)
ax1.set_xlim(START_YEAR, END_YEAR)

# ── Gráfico 2: Calentamiento por ciudad ──
ax2 = fig.add_subplot(gs[1, 0])
cities_sorted = sorted(warming, key=warming.get, reverse=True)
values = [warming[c] for c in cities_sorted]
colors = [palette[c] for c in cities_sorted]
bars = ax2.barh(cities_sorted, values, color=colors, edgecolor="white", height=0.6)
for bar, val in zip(bars, values):
    ax2.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
             f"+{val}°C", va="center", fontsize=11, fontweight="bold")
ax2.set_title(f"Calentamiento total\n({first_decade} → {last_decade})", fontsize=13, pad=10)
ax2.set_xlabel("°C")
ax2.set_xlim(0, max(values) + 0.6)
ax2.invert_yaxis()

# ── Gráfico 3: Temperatura media por mes (boxplot) ──
ax3 = fig.add_subplot(gs[1, 1])
monthly_avg = (
    raw.groupby(["city", "month"])["temp"]
    .mean()
    .reset_index()
)
month_names = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]
for city in CITIES:
    sub = monthly_avg[monthly_avg["city"] == city].sort_values("month")
    ax3.plot(range(1, 13), sub["temp"], marker="o", markersize=4,
             color=palette[city], linewidth=1.8, label=city)
ax3.set_title("Temperatura media mensual\n(serie completa)", fontsize=13, pad=10)
ax3.set_xticks(range(1, 13))
ax3.set_xticklabels(month_names)
ax3.set_ylabel("Temperatura (°C)")
ax3.legend(fontsize=9, framealpha=0.9)

plt.savefig("climate_spain_analysis.png", dpi=150, bbox_inches="tight",
            facecolor="white", edgecolor="none")
print("\n✅ Gráfico guardado como: climate_spain_analysis.png")
plt.show()

# ─────────────────────────────────────────────
# 5. RESUMEN FINAL
# ─────────────────────────────────────────────

print("\n" + "=" * 55)
print("  CONCLUSIONES")
print("=" * 55)
most_warmed = max(warming, key=warming.get)
least_warmed = min(warming, key=warming.get)
avg_warming = round(sum(warming.values()) / len(warming), 2)
print(f"""
  • Las 5 ciudades analizadas muestran una tendencia
    clara de calentamiento entre 1950 y 2024.

  • Ciudad con mayor calentamiento : {most_warmed} (+{warming[most_warmed]}°C)
  • Ciudad con menor calentamiento : {least_warmed} (+{warming[least_warmed]}°C)
  • Calentamiento medio del conjunto: +{avg_warming}°C

  • La tendencia es consistente con los datos del
    cambio climático a escala europea.
""")
print("=" * 55)
