# analyze_submission.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def analyze_submission(file_path: str):
    """
    Проводит детальный анализ сгенерированного submission-файла.
    """
    try:
        df = pd.read_csv(file_path, header=None)
        df.columns = ['E_mu_Z', 'R', 's', 'p']
    except FileNotFoundError:
        print(f"Ошибка: Файл {file_path} не найден.")
        return
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")
        return

    print("=" * 50)
    print(f"Анализ файла: {file_path}")
    print("=" * 50)

    # 1. Общая информация
    print("\n[1] Общая информация:")
    print(f"  - Всего строк: {len(df)}")
    
    # Проверка консистентности
    s_plus_p_check = (df['s'] + df['p'] == 4800).all()
    print(f"  - Проверка s + p = 4800: {'✅ OK' if s_plus_p_check else '❌ ОШИБКА'}")
    if not s_plus_p_check:
        print(f"    Пример неверной суммы: s={df.iloc[0]['s']}, p={df.iloc[0]['p']}, сумма={df.iloc[0]['s'] + df.iloc[0]['p']}")

    # 2. Анализ предсказанных R
    print("\n[2] Анализ скоростей кода (R):")
    r_counts = df['R'].value_counts().sort_index()
    r_proportions = df['R'].value_counts(normalize=True).sort_index()
    
    print("  - Распределение по классам R:")
    for r_val, count in r_counts.items():
        proportion = r_proportions[r_val] * 100
        print(f"    R = {r_val:.2f}  : {count:4d} строк ({proportion:5.2f}%)")

    # 3. Анализ предсказанных s и p
    print("\n[3] Статистики для сокращенных (s) и проколотых (p) узлов:")
    print("  - Статистики для 's':")
    print(df['s'].describe().to_string())
    print("\n  - Статистики для 'p':")
    print(df['p'].describe().to_string())

    # 4. Анализ E_mu_Z
    print("\n[4] Статистики для параметра E_mu_Z:")
    print(df['E_mu_Z'].describe().to_string())
    
    print("\n" + "=" * 50)

    # 5. Визуализация
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Анализ Submission-файла', fontsize=16)

    # Распределение R
    sns.countplot(x='R', data=df, ax=axes[0, 0], order=sorted(df['R'].unique()))
    axes[0, 0].set_title('Распределение предсказанных R')
    axes[0, 0].set_xlabel('Скорость кода (R)')
    axes[0, 0].set_ylabel('Количество')

    # Распределение s
    sns.histplot(df['s'], kde=True, ax=axes[0, 1])
    axes[0, 1].set_title('Распределение числа сокращенных узлов (s)')
    axes[0, 1].set_xlabel('Число узлов (s)')
    axes[0, 1].set_ylabel('Частота')

    # Распределение E_mu_Z
    sns.histplot(df['E_mu_Z'], kde=True, ax=axes[1, 0])
    axes[1, 0].set_title('Распределение параметра E_mu_Z')
    axes[1, 0].set_xlabel('E_mu_Z')
    axes[1, 0].set_ylabel('Частота')
    
    # Boxplot s для каждого R
    sns.boxplot(x='R', y='s', data=df, ax=axes[1, 1], order=sorted(df['R'].unique()))
    axes[1, 1].set_title('Распределение s для каждого предсказанного R')
    axes[1, 1].set_xlabel('Скорость кода (R)')
    axes[1, 1].set_ylabel('Число узлов (s)')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze a submission.csv file.")
    parser.add_argument("file_path", type=str, nargs='?', default="submission.csv", 
                        help="Path to the submission file to analyze. Defaults to 'submission.csv'.")
    args = parser.parse_args()
    analyze_submission(args.file_path)