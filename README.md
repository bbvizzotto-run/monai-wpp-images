# Robustness of AI-Based Medical Imaging Under Real-World Image Compression: A MONAI Study on WhatsApp-Shared Radiographs

Este repositório contém o código-fonte e os recursos de suporte para o artigo "Robustness of AI-Based Medical Imaging Under Real-World Image Compression: A MONAI Study on WhatsApp-Shared Radiographs", submetido ao periódico *Discover Artificial Intelligence*.

## Visão Geral

Este estudo investiga a robustez do desempenho diagnóstico assistido por IA em imagens médicas sob condições realistas de compressão de imagem, especificamente em radiografias panorâmicas compartilhadas via WhatsApp. Utilizamos um pipeline de deep learning baseado em MONAI para avaliar como a compressão de imagem com perdas afeta a confiabilidade diagnóstica, comparando o desempenho de avaliadores humanos e modelos de IA.

## Estrutura do Repositório

*   `data/`: Contém informações sobre os datasets utilizados (não os dados brutos devido a questões de privacidade e tamanho).
*   `src/`: Contém o código-fonte para o pipeline MONAI, treinamento do modelo e scripts de avaliação.
    *   `src/monai_pipeline.py`: Script principal do pipeline MONAI.
    *   `src/evaluation_scripts.py`: Scripts para avaliação do desempenho diagnóstico (humano e IA).
*   `notebooks/`: Jupyter notebooks para exploração de dados, visualização de resultados e análises adicionais.
*   `models/`: Modelos pré-treinados (se aplicável e permitido).
*   `results/`: Resultados da avaliação, incluindo métricas e visualizações.

## Como Usar

1.  **Clonar o Repositório:**
    ```bash
    git clone https://github.com/bbvizzotto-run/Discover_AI_Robustness_Study.git
    cd Discover_AI_Robustness_Study
    ```
2.  **Configurar o Ambiente:**
    Instale as dependências necessárias (Python 3.x, PyTorch, MONAI, etc.). Um arquivo `requirements.txt` será fornecido na pasta `src/`.
3.  **Executar o Pipeline MONAI:**
    Siga as instruções no `src/monai_pipeline.py` para treinar ou inferir com o modelo.
4.  **Realizar Avaliações:**
    Utilize os scripts em `src/evaluation_scripts.py` para reproduzir as análises de desempenho.

## Licença

Este projeto está licenciado sob a Licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

## Citação

Se você usar este código ou dados em sua pesquisa, por favor, cite nosso artigo:

```bibtex
@article{vizzotto202Xrobustness,
  title={Robustness of AI-Based Medical Imaging Under Real-World Image Compression: A MONAI Study on WhatsApp-Shared Radiographs},
  author={Vizzotto, Bruno Boessio and Vizzotto, Mariana Boessio},
  journal={Discover Artificial Intelligence},
  year={202X},
  doi={Aguardando DOI}
}
```

## Contato

Para dúvidas ou sugestões, entre em contato com Bruno Boessio Vizzotto (bbvizzotto@ufg.br).
