# Caixa d'Enginyers Heatmap Demo 🌍

Mapa de calor interactivo que identifica las zonas con mayor potencial de apertura de oficinas
o puntos móviles de Caixa d’Enginyers, equilibrando impacto social y sostenibilidad económica.

Interesting SOTA:
https://www.perplexity.ai/search/i-am-a-data-scientist-competin-dQMr4RjaQzKpvCJNtuaKFw#0

## 🚀 Ejecutar localmente
```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```
## Datasets
- [data/population.csv](data/population.csv): Población por municipio (total/hombres/mujeres) en 2023.
	- NAMEUNIT: Nombre del municipio.
	- POB21: Población total.
	- HOMBRES: Población masculina.
	- MUJERES: Población femenina.
	- Densidad: Densidad de población (hab/km2).
	- Superficie_km2: Superficie del municipio en km2.
- [data/age_population.csv](data/age_population.csv): Población por municipio y por edades en 2018.
	- PAD_2_MU_2018_PAD_2C02: Población total.
	- PAD_2_MU_2018_PAD_2C**XX**: Población por rango de edad. **XX** va de 03 a 20, cada uno es un rango de 5 años, de 0-4 años hasta 85 y más.
- [data/rent.csv](data/rent.csv): Precio de alquiler por distrito de municipio (hacer average) en 2024.
	- NMUN: Nombre del municipio (multiples distritos por municipio, hacer media).
	- Renta_Medi: Mediana de la renta mensual en euros por metro cuadrado - vivienda colectiva.
	- Renta_Me_1: Mediana de la renta mensual en euros por metro cuadrado - vivienda unifamiliar.
- [data/household_municipality.csv](data/household_municipality.csv): Renta por municipio en 2023.
	- Name: Nombre del municipio. Formato: "Terrassa-08279"
	- RENTA BRUTA MEDIA: Renta bruta media anual por hogar en euros.
- [data/banks-by-population.csv](data/banks-by-population.csv): Número de oficinas bancarias por municipio en 2023.
	- Municipality: Nombre del municipio.
	- Banks: Número de oficinas bancarias.
	- Longitude
	- Latitude
