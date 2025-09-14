#!/usr/bin/env python3
"""
Script de prueba para verificar las correcciones aplicadas
"""

import sys
import os

def test_block_matches_fix():
    """Verifica que la estructura de block_matches sea correcta"""
    print("🔍 Verificando corrección de block_matches...")
    
    # Importar las funciones relevantes
    try:
        from copy_move_detection import detect_copy_move_blocks
        from visualization import create_block_match_visualization
        import numpy as np
        from PIL import Image
        import matplotlib.pyplot as plt
        
        # Crear una imagen de prueba simple
        test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        test_pil = Image.fromarray(test_image)
        
        # Simular resultados con la estructura correcta
        test_results = {
            'block_matches': [
                {
                    'block1_pos': (10, 10),
                    'block2_pos': (50, 50),
                    'correlation': 0.95,
                    'distance': 40.0
                },
                {
                    'block1_pos': (20, 20),
                    'block2_pos': (60, 60),
                    'correlation': 0.92,
                    'distance': 45.0
                }
            ]
        }
        
        # Intentar crear la visualización
        fig, ax = plt.subplots(figsize=(8, 6))
        create_block_match_visualization(ax, test_pil, test_results)
        plt.close(fig)
        
        print("✅ Corrección de block_matches funcionando correctamente")
        return True
        
    except KeyError as e:
        print(f"❌ Error KeyError detectado: {e}")
        print("   La estructura de block_matches aún no es compatible")
        return False
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        return False

def test_import_modules():
    """Verifica que todos los módulos se importen correctamente"""
    print("\n🔍 Verificando importación de módulos...")
    
    modules_to_test = [
        'app',
        'visualization',
        'copy_move_detection',
        'main',
        'utils',
        'export_utils'
    ]
    
    all_ok = True
    for module in modules_to_test:
        try:
            __import__(module)
            print(f"  ✅ {module} importado correctamente")
        except ImportError as e:
            print(f"  ❌ Error al importar {module}: {e}")
            all_ok = False
        except Exception as e:
            print(f"  ⚠️ Advertencia en {module}: {e}")
    
    return all_ok

def main():
    """Función principal de prueba"""
    print("=" * 60)
    print("PRUEBA DE CORRECCIONES APLICADAS")
    print("=" * 60)
    
    results = []
    
    # Prueba 1: Importación de módulos
    results.append(("Importación de módulos", test_import_modules()))
    
    # Prueba 2: Corrección de block_matches
    results.append(("Corrección block_matches", test_block_matches_fix()))
    
    # Resumen
    print("\n" + "=" * 60)
    print("RESUMEN DE PRUEBAS")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASÓ" if result else "❌ FALLÓ"
        print(f"{test_name}: {status}")
    
    print(f"\nResultado final: {passed}/{total} pruebas pasadas")
    
    if passed == total:
        print("\n🎉 ¡Todas las correcciones funcionan correctamente!")
        return 0
    else:
        print(f"\n⚠️ Algunas pruebas fallaron. Por favor revise los errores.")
        return 1

if __name__ == "__main__":
    sys.exit(main())