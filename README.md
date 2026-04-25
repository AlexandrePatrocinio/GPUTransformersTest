# Passo a Passo Detalhado para Criar um Projeto Python com CUDA para o Modelo Qwen2.5-Coder-7B-Instruct

Este é um guia completo para iniciar um novo projeto Python do zero. Este guia assume um ambiente Windows, mas pode ser adaptado para outros sistemas. O foco é garantir suporte a CUDA para aceleração de GPU, evitando erros de importação e incompatibilidades de versão.

## Pré-requisitos CUDA 12.1

- **Python 3.12:** (ou superior, mas teste a compatibilidade com PyTorch + CUDA 12.1). Nos testes para criar este guia a versão `Python 3.12` foi suficiente para executar com o `CUDA 12.1`. Baixe e instale do site oficial: https://www.python.org/downloads/ ou use seu instalador preferido (Ex: apt no linux ou winget no windows).
- **CUDA Toolkit:** A placa de vídeo usada no teste suporta CUDA 13.0 (`NVIDIA RTX 500 Ada Generation`) mas pôde executar a versão `CUDA 12.1` sem problemas. `Importante:` a versão compatível do `PyTorch` foi a `2.5.1`. O torch é a biblioteca que usa o CUDA para conseguir rodar o modelo LLM testado, portanto precisam ser compatíveis. Baixe do site da NVIDIA: https://developer.nvidia.com/cuda-downloads. Verifique se sua GPU suporta CUDA (ex.: NVIDIA GeForce RTX ou superior).

## Pré-requisitos CUDA 13.0

- **Python 3.14:** (a versão mais recente no momento da criação deste guia). Não houveram problemas em usar o `CUDA 13.0` com a última versão do Python neste guia. Você poderá baixá-lo normalmente do site oficial: https://www.python.org/downloads/ ou use seu instalador preferido (Ex: apt no linux ou winget no windows).
- **CUDA Toolkit:** O `CUDA 13.0` é a versão nativa disponível para a placa de vídeo deste guia (`NVIDIA RTX 500 Ada Generation`). Desta forma a versão compatível da dependência `PyTorch` testada foi a `2.11.0`(versão estável mais recente no momento da criação deste guia). Baixe do site da NVIDIA: https://developer.nvidia.com/cuda-downloads. Verifique se sua GPU suporta CUDA (ex.: NVIDIA GeForce RTX ou superior).

## Pré-requisitos opcionais
- **Git** (opcional, para clonar repositórios se necessário).
- Uma conta no `Hugging Face` (gratuita) para acessar modelos: https://huggingface.co/join. Configure um token de acesso (HF_TOKEN) para downloads mais rápidos e sem limites.

## Passo 1: Criar e Configurar o Ambiente Virtual

1. Abra o Prompt de Comando ou PowerShell como administrador.
2. Navegue até o diretório onde deseja criar o projeto (ex.: `cd C:\Users\SeuUsuario\Documents\Projetos`).
3. Crie uma nova pasta para o projeto: `mkdir ProjetoQwenCUDA`.
4. Entre na pasta: `cd ProjetoQwenCUDA`.
5. Crie um ambiente virtual com a versão do Python escolhido (`3.12` ou `3.14`). Exemplo: `python -m venv .venv312` (o nome `".venv312"` indica `Python 3.12`).
6. Ative o ambiente virtual. Por exemplo com o `.venv312` do exemplo acima será: `.venv312\Scripts\Activate.ps1`. Você verá `(.venv312)` no prompt.
7. Atualize o pip para a versão mais recente: `python -m pip install --upgrade pip`.
8. Com o python escolhido já instalado configure o ambiente para a sua versão correta. Exemplo com o `Python 3.14`: `py -3.14 -m venv .venv314`. Caso ainda não tenha instalado a versão escolhida então o instale como já indicado na sessão ## Pré-requisitos.

## Passo 2: Instalar Dependências com Suporte a CUDA

1. Crie os arquivos `requirements312.txt` (para Python `3.12`) ou `requirements314.txt` (para Python `3.14`) na raiz do projeto com o seguinte conteúdo (versões específicas para evitar incompatibilidades):

   - `requirements312.txt`
   ```
   transformers==5.5.3
   accelerate
   safetensors
   bitsandbytes==0.49.2
   ```

   - `requirements314.txt`
   ```
   transformers==5.5.4
   accelerate
   safetensors
   bitsandbytes==0.49.2
   ```

   - Salve o arquivo.

2. Instale o pacote torch via pip, especificando o índice PyTorch para `CUDA 12.1` ou `CUDA 13.0` (escolha conforme seu caso):
   ```
   pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
   ```
   - Isso instala PyTorch com suporte a CUDA (torch 2.5.1+cu121).

   ```
   pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
   ```
   - Isso instala PyTorch com suporte a CUDA (torch 2.11.0+cu130).   

3. Instale as outras dependências especificadas no arquivo requirementsXXX.txt (onde XXX é a sua versão do Python):
   ```
   pip install -r requirementsXXX.txt
   ```
   - Isso instala transformers (compatível), accelerate para otimização de GPU e bitsandbytes para a quantização dos modelos.

4. Verifique as instalações:
   - `pip show torch` (deve mostrar versão = `2.5.1+cu121` ou `2.11.0+cu130`).
   - `pip show transformers` (deve ser = `5.5.3` ou `5.5.4`).
   - `pip show bitsandbytes` (deve ser = `0.49.2`).
   - Teste CUDA: Execute `python -c "import torch; print(torch.cuda.is_available())"`. Deve retornar `True` se CUDA estiver funcionando.

## Passo 3: Baixar e Configurar o Modelo

1. Crie uma pasta para cache do modelo: `mkdir Models\Coders\Qwen`.
2. O modelo Qwen2.5-Coder-7B-Instruct será baixado automaticamente na primeira execução, mas você pode pré-baixá-lo para acelerar:
   - Execute temporariamente: `python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-Coder-7B-Instruct', cache_dir='Models/Coders/Qwen')"` (pode levar tempo na primeira vez).
3. A versão quantizada do modelo `Qwen2.5-Coder (Qwen/Qwen2.5-Coder-7B-Instruct-AWQ)` não é suportada na versão do Torch com `CUDA 12.1`.
   Na verdade, ele precisa da dependência `gptqmodel` instalada que não é compatível com o `torch 2.5.1`.
4. Na utilização da versão do `torch 2.11.0` com `CUDA 13.0`, foi usado uma outra abordagem para os modelos AWQ mas igualmente sem sucesso.
   Neste caso a biblioteca `autoawq` foi utilizada mas a instalação falhou por falta de compatibilidade com a nova versão do `Python 3.14`
   - Conclusão, modelos AWQ (menores em tamanho) não serão possíveis nesta versão do projeto.

## Passo 4: Criar o Script Principal

1. Crie um arquivo `main.py` na raiz do projeto. Basei-se no Test.py presente na raiz deste prrepositório git:
2. Salve o arquivo.

## Passo 5: Executar e Testar o Projeto

1. Certifique-se de que o ambiente virtual está ativado. Exemplo com o `Python 3.12`: `.venv312\Scripts\Activate.ps1`.
2. Execute o script: `python main.py`.
   - Se o modelo ainda não foi baixado, então na primeira execução ele será baixado (pode levar 10-30 minutos dependendo da internet/GPU).
   - Você verá progresso de download e carregamento.
   - Warnings sobre HF_TOKEN ou dispositivo são normais e não impedem a execução.
3. Verifique a saída: Deve imprimir o prompt e o texto gerado pelo modelo.
4. Se houver erros:
   - Verifique CUDA: `python -c "import torch; print(torch.cuda.get_device_name(0))"`.
   - Reinstale pacotes se necessário.
   - Para modelos grandes, certifique-se de ter pelo menos 16GB RAM GPU. Nos testes deste guia a `NVIDIA RTX 500 Ada Generation` possui apenas 4GB VRAM para a GPU. Portanto, extremamente lento. É possivel melhorar a performance como descrito no tópico abaixo.

## Dicas Adicionais e Troubleshooting

### Desempenho
No arquivo Test.py deste projeto você encontrará duas variáveis: `device` e `device_map`. `device` aceita os valores `cpu` que, evidentemente, executa o modelo na CPU e `gpu` para a execução na GPU. Porém `device_map`, possui também `auto` além das opções `cpu` e `gpu` para o mapeamento e carregamento do modelo em memória. Use `device_map="auto"` para distribuir o modelo entre GPU/CPU quando ele é muito grande usando CUDA. Uma ressalva aqui é o swap dos dados em memória. A troca constante do modelo entre a VRAM da GPU (bem mais rápida) e a RAM da CPU (mais lenta) torna a execução bem mais lenta em média. Isso em comparação à carregá-la totalmente na VRAM da GPU ou totalmente na RAM da CPU. Nos testes mostrados mais abaixo a opção `device_map="auto"` não foi usada mas apenas `cuda`. O modelo não quantizado, por ser maior, carregará apenas as camadas suportadas pelo limite da VRAM (4GB no caso deste guia). O restante automaticamente ficará na RAM.

Para GPUs com pouca memória VRAM, considere quantização (ex.: AWQ) porém outras versões do torch e transformers serão necessárias. Existem ainda modelos menores como o `Qwen2.5-Coder-3B-Instruct` de aproximadamente 6GB e o `Qwen/Qwen2.5-Coder-1.5B-Instruct` de 3.09GB. Neste caso o trade-off é trocar capacidade de "raciocínio" em codificações mais complexas inerente ao modelo de 7 bilhoes de parâmentros pela econômia de memória e desempenho de execução innerentes ao modelo de 3 bilhões de parâmetros por exemplo.

O uso da quantização, no caso deste guia, tornou-se obrigatória pelo hardware limitado. Este recurso diminui drasticamente o tempo de resposta do modelo. Para tanto, usou-se a biblioteca `bitsandbytes` com a opção `bnb_4bit_use_double_quant=True` que ajudou a contornar o problema realizando a quantização em tempo de carregamento. O modelo diminuiu de 15.23GB para aproximadamente 4.0GB segundo meus testes. O trade-off aqui é que trocamos a precisão de calculo (float 16 bits para NormalFloat 4 bits) por performance. Veja testes mais abaixo neste documento.

Isto permite cenarios de testes ou provas de conceito mas ainda muito lento para aplicações reais. Para aplicações reais é preciso levar em conta ainda a janela de contexto. O uso real de memória neste caso, aumentará conforme o tamanho do contexto (tokens de entrada/saída) processado.

### Erros Comuns

- **ImportError com torch**: Atualize torch para >=2.5.1 via o índice CUDA.
  ```
  pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu121
  ```

- **CUDA não disponível**: Reinstale CUDA Toolkit e verifique drivers NVIDIA.
  ```
  python -c "import torch; print(torch.cuda.is_available())"
  ```

- **Download lento**: Use HF_TOKEN e uma conexão estável.

### Limpeza
Para desativar o venv, digite `.venv312\Scripts\deactivate`.

---

## Resumo da Estrutura do Projeto deste guia

### com CUDA 12.1

```
ProjetoQwenCUDA/
├── .venv312/               # Ambiente virtual
├── Models/
│   └── Coders/
│       └── Qwen/           # Cache dos modelos
├── requirements312.txt     # Dependências do projeto
└── main.py                 # Script principal
```

### com CUDA 13.0

```
ProjetoQwenCUDA/
├── .venv314/               # Ambiente virtual
├── Models/
│   └── Coders/
│       └── Qwen/           # Cache dos modelos
├── requirements314.txt     # Dependências do projeto
└── main.py                 # Script principal
```


# Testes

## Setup
- i7 ultra
- 64 GB DDR5
- SSD NVMe 1TB
- NVIDIA RTX 500 Ada Generation Laptop GPU 4GB VRAM GDDR6

## Todos os testes usam o mesmo prompt
**Prompt:** Create a minimal api in C# dotnet 10 with an endpoint that returns a JSON response. It should have a single endpoint at /api/hello that returns { "message": "Hello, World!" } but without using any external libraries or controllers.

**LLM:** Qwen/Qwen2.5-Coder-7B-Instruct

**Tamanho total:** 15.23 GB

### Testes Comparativos

**Dependências:** Python 3.12.10, torch 2.5.1, CUDA 12.1, transformers 5.5.3 e bitsandbytes 0.49.2

| Testes  | Device      | Quantização (4 bits) | Tempo total de resposta |
|---------|-------------|----------------------|-------------------------|
| Teste 1 | CUDA        | Não                  | 10:52 min               |
| Teste 2 | CPU         | Não                  | 38:37 min               |
| Teste 3 | CUDA        | Sim                  | 1:38 min                |
| Teste 4 | CPU         | Sim                  | ??? infinity loop       |

**Dependências:** Python 3.14.3, torch 2.11.0, CUDA 13.0, transformers 5.5.4 e bitsandbytes 0.49.2

| Testes  | Device      | Quantização (4 bits) | Tempo total de resposta |
|---------|-------------|----------------------|-------------------------|
| Teste 1 | CUDA        | Não                  | 6:25 min                |
| Teste 2 | CPU         | Não                  | 1:54 min                |
| Teste 3 | CUDA        | Sim                  | 0:51 min                |
| Teste 4 | CPU         | Sim                  | ??? infinity loop       |

### Exemplo de resultado

Creating a minimal API in .NET Core (now known as .NET) involves using the `WebApplication` class to configure and run your application. Below is an example of how you can create a minimal API with a single endpoint `/api/hello` that returns a JSON response `{ "message": "Hello, World!" }`.

First, ensure you have .NET installed on your machine. Then, you can create a new project using the following command:

```sh
dotnet new web -n MinimalApiExample
cd MinimalApiExample
```

Next, open the `Program.cs` file and replace its content with the following code:

```csharp
using Microsoft.AspNetCore.Builder;
using Microsoft.Extensions.DependencyInjection;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

var app = builder.Build();

// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.MapGet("/api/hello", () =>
{
    return new { message = "Hello, World!" };
});

app.Run();
```

This code sets up a basic ASP.NET Core application with a single GET endpoint `/api/hello`. When this endpoint is accessed, it returns a JSON object containing a message.

To run the application, use the following command:

```sh
dotnet run
```

Once the application is running, you can access the endpoint by navigating to `https://localhost:5001/api/hello` (or `http://localhost:5000/api/hello` if you're using HTTP) in your web browser or using a tool like Postman. You should see the JSON response:

```json
{
  "message": "Hello, World!"
}
```

This demonstrates how to create a minimal API in .NET Core with a single endpoint that returns a JSON response without using any external libraries or controllers.
Elapsed time: 1 min 38 s
