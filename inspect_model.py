#!/usr/bin/env python3
"""
Script para inspeccionar el modelo optimizado
"""
import pickle
import numpy as np

print("Inspeccionando modelo optimizado...")

with open('models/optimized_har_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

print("\nContenido del modelo:")
for key in model_data.keys():
    print(f"  - {key}")

print(f"\nInformacion del Feature Selector:")
if 'feature_selector' in model_data:
    selector = model_data['feature_selector']
    print(f"  - Numero de caracteristicas seleccionadas: {selector.k}")
    print(f"  - Total caracteristicas originales: {len(selector.get_support())}")

    selected_indices = np.where(selector.get_support())[0]
    print(f"  - Indices de caracteristicas seleccionadas: {selected_indices}")

print(f"\nNombres de caracteristicas:")
if 'feature_names' in model_data:
    feature_names = model_data['feature_names']
    print(f"  - Total: {len(feature_names)}")
    for i, name in enumerate(feature_names):
        print(f"    {i}: {name}")

print(f"\nCaracteristicas seleccionadas finales:")
if 'feature_selector' in model_data and 'feature_names' in model_data:
    selector = model_data['feature_selector']
    feature_names = model_data['feature_names']
    selected_mask = selector.get_support()

    selected_features = [name for name, selected in zip(feature_names, selected_mask) if selected]
    print(f"  - Cantidad: {len(selected_features)}")
    for i, name in enumerate(selected_features):
        print(f"    {i}: {name}")

print(f"\nInformacion del Scaler:")
if 'scaler' in model_data:
    scaler = model_data['scaler']
    print(f"  - Numero de caracteristicas esperadas: {scaler.n_features_in_}")
    print(f"  - Shape de mean: {scaler.mean_.shape}")
    print(f"  - Shape de scale: {scaler.scale_.shape}")

print(f"\nResultados del entrenamiento:")
if 'results' in model_data:
    results = model_data['results']
    print(f"  - Baseline accuracy: {results.get('baseline_accuracy', 'N/A')}")
    print(f"  - Final accuracy: {results.get('final_accuracy', 'N/A')}")
    print(f"  - Improvement: {results.get('improvement', 'N/A')}")
