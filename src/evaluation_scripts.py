import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_curve, auc, confusion_matrix
from statsmodels.stats.contingency_tables import mcnemar
import matplotlib.pyplot as plt
import argparse
import os

def calculate_metrics(y_true, y_pred_proba, threshold=0.5):
    """Calcula métricas de desempenho para classificação binária."""
    y_pred = (y_pred_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    accuracy = accuracy_score(y_true, y_pred)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    kappa = cohen_kappa_score(y_true, y_pred)

    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    return {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'kappa': kappa,
        'roc_auc': roc_auc,
        'fpr': fpr,
        'tpr': tpr
    }

def run_mcnemar_test(y_true, y_pred1, y_pred2, threshold=0.5):
    """Executa o teste de McNemar para comparar duas previsões correlacionadas."""
    # Converte probabilidades para classes binárias
    y_pred1_binary = (y_pred1 >= threshold).astype(int)
    y_pred2_binary = (y_pred2 >= threshold).astype(int)

    # Cria a tabela de contingência 2x2 para McNemar
    # a: ambos corretos
    # b: pred1 incorreto, pred2 correto
    # c: pred1 correto, pred2 incorreto
    # d: ambos incorretos

    # Casos onde a previsão 1 está correta e a previsão 2 está incorreta
    b = np.sum((y_true == y_pred1_binary) & (y_true != y_pred2_binary))
    # Casos onde a previsão 1 está incorreta e a previsão 2 está correta
    c = np.sum((y_true != y_pred1_binary) & (y_true == y_pred2_binary))

    # A tabela para mcnemar é [[n00, n01], [n10, n11]]
    # n01 = b (pred1 erra, pred2 acerta)
    # n10 = c (pred1 acerta, pred2 erra)
    # Os outros elementos não são usados no teste de McNemar para significância
    # mas a função espera uma tabela 2x2 completa.
    # Para simplificar, podemos usar apenas b e c para o teste de McNemar.
    # statsmodels.stats.contingency_tables.mcnemar espera a tabela na forma:
    # [[n_00, n_01], [n_10, n_11]]
    # onde n_01 são os casos onde o método 1 erra e o método 2 acerta
    # e n_10 são os casos onde o método 1 acerta e o método 2 erra

    # Construindo a tabela de discordância
    table = [[0, b], [c, 0]] # n00 e n11 não são relevantes para o p-value do McNemar

    result = mcnemar(table, exact=True)
    return result.pvalue

def plot_roc_curve(fpr, tpr, roc_auc, title, filename):
    """Plota e salva a curva ROC."""
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(filename)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Script de avaliação de desempenho diagnóstico.")
    parser.add_argument('--true_labels', type=str, required=True, help='Caminho para o arquivo CSV com os rótulos verdadeiros.')
    parser.add_argument('--predictions_original', type=str, required=True, help='Caminho para o arquivo CSV com as probabilidades de previsão do modelo original.')
    parser.add_argument('--predictions_whatsapp', type=str, required=True, help='Caminho para o arquivo CSV com as probabilidades de previsão do modelo WhatsApp.')
    parser.add_argument('--output_dir', type=str, default='./results', help='Diretório para salvar os resultados e gráficos.')
    parser.add_argument('--threshold', type=float, default=0.5, help='Limiar de classificação para converter probabilidades em classes binárias.')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Carregar dados
    y_true = pd.read_csv(args.true_labels).values.flatten()
    y_pred_proba_original = pd.read_csv(args.predictions_original).values.flatten()
    y_pred_proba_whatsapp = pd.read_csv(args.predictions_whatsapp).values.flatten()

    # Validar se os tamanhos dos arrays são iguais
    if not (len(y_true) == len(y_pred_proba_original) == len(y_pred_proba_whatsapp)):
        raise ValueError("Todos os arquivos de entrada devem ter o mesmo número de amostras.")

    # Avaliar o modelo original
    metrics_original = calculate_metrics(y_true, y_pred_proba_original, args.threshold)
    print("\n--- Métricas de Desempenho (Modelo Original) ---")
    for metric, value in metrics_original.items():
        if metric not in ['fpr', 'tpr']:
            print(f"{metric.replace('_', ' ').title()}: {value:.4f}")
    plot_roc_curve(metrics_original['fpr'], metrics_original['tpr'], metrics_original['roc_auc'],
                   'Curva ROC - Modelo Original', os.path.join(args.output_dir, 'roc_curve_original.png'))

    # Avaliar o modelo WhatsApp
    metrics_whatsapp = calculate_metrics(y_true, y_pred_proba_whatsapp, args.threshold)
    print("\n--- Métricas de Desempenho (Modelo WhatsApp) ---")
    for metric, value in metrics_whatsapp.items():
        if metric not in ['fpr', 'tpr']:
            print(f"{metric.replace('_', ' ').title()}: {value:.4f}")
    plot_roc_curve(metrics_whatsapp['fpr'], metrics_whatsapp['tpr'], metrics_whatsapp['roc_auc'],
                   'Curva ROC - Modelo WhatsApp', os.path.join(args.output_dir, 'roc_curve_whatsapp.png'))

    # Executar Teste de McNemar
    mcnemar_p_value = run_mcnemar_test(y_true, y_pred_proba_original, y_pred_proba_whatsapp, args.threshold)
    print(f"\n--- Teste de McNemar (p-value): {mcnemar_p_value:.4f} ---")
    if mcnemar_p_value < 0.05:
        print("Há uma diferença estatisticamente significativa entre os dois modelos.")
    else:
        print("Não há diferença estatisticamente significativa entre os dois modelos.")

    # Salvar métricas em um arquivo de texto
    with open(os.path.join(args.output_dir, 'evaluation_results.txt'), 'w') as f:
        f.write("--- Métricas de Desempenho (Modelo Original) ---\n")
        for metric, value in metrics_original.items():
            if metric not in ['fpr', 'tpr']:
                f.write(f"{metric.replace('_', ' ').title()}: {value:.4f}\n")
        f.write("\n--- Métricas de Desempenho (Modelo WhatsApp) ---\n")
        for metric, value in metrics_whatsapp.items():
            if metric not in ['fpr', 'tpr']:
                f.write(f"{metric.replace('_', ' ').title()}: {value:.4f}\n")
        f.write(f"\n--- Teste de McNemar (p-value): {mcnemar_p_value:.4f} ---\n")
        if mcnemar_p_value < 0.05:
            f.write("Há uma diferença estatisticamente significativa entre os dois modelos.\n")
        else:
            f.write("Não há diferença estatisticamente significativa entre os dois modelos.\n")

if __name__ == "__main__":
    main()
