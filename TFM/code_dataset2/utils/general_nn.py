import torch
import os
import pandas as pd


def guardar_resultados(modelo, name_modelo, problem, resultados_mejor_modelo, results_path = "./results",path_save_experiment=None, params=None):
    
    print("\n==================== GUARDANDO RESULTADOS ===================\n")
    path_model = results_path+f"/{problem}"+ f"/{name_modelo}.pt"
    torch.save(modelo.state_dict(), path_model)


    resultados_general_save = {"Modelo": name_modelo, 
                            "Params": params_gt, 
                           "Fichero_resultados_experimento": path_save_experiment, 
                            "Loss_tst": resultados_mejor_modelo["loss_test"],
                            "R2_tst": resultados_mejor_modelo["r2_test"],
                            "Loss_nodes": resultados_mejor_modelo["loss_nodes"],
                            "R2_eval": resultados_mejor_modelo["r2_eval_final"],
                            "Loss_eval": resultados_mejor_modelo["loss_eval_final"],
                            "Loss_final": resultados_mejor_modelo["loss_final"]
                            }


    path_general_problem = results_path+f"/{problem}"+ "/results.csv"

    if os.path.exists(path_general_problem):
        df = pd.read_csv(path_general_problem)
    else:
        # Crear un DataFrame vacío con las columnas del diccionario
        df = pd.DataFrame(columns=resultados_general_save.keys())

    new_data_df = pd.DataFrame([resultados_general_save])

    df = pd.concat([df, new_data_df], ignore_index=True)
    print(df)
    df.to_csv(path_general_problem, index=False)

    print("\n==================== RESULTADOS GUARDADOS ===================\n")

