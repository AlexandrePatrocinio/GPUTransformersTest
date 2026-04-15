# Passo a Passo Detalhado para Criar um Projeto Python com CUDA para o Modelo Qwen2.5-Coder-7B-Instruct

Este é um guia completo para iniciar um novo projeto Python do zero. Este guia assume um ambiente Windows, mas pode ser adaptado para outros sistemas. O foco é garantir suporte a CUDA para aceleração de GPU, evitando erros de importação e incompatibilidades de versão.

## Pré-requisitos

- **Python 3.12** (ou superior, mas teste a compatibilidade com PyTorch CUDA). Nos testes para criar este guia o único compatível foi o Python 3.12. Baixe e instale do site oficial: https://www.python.org/downloads/ ou use seu instalador preferido (Ex: apt no linux ou winget no windows).
- **CUDA Toolkit** Apesar da placa de vídeo usada no teste aceitar CUDA 13.0 (NVIDIA RTX 500 Ada Generation), a versão CUDA 12.1 foi a versão compatível com o PyTorch 2.5.1. Para conseguir rodar o modelo LLM testado, essa foi a versão do Torch que funcionou. Baixe do site da NVIDIA: https://developer.nvidia.com/cuda-downloads. Verifique se sua GPU suporta CUDA (ex.: NVIDIA GeForce RTX ou superior).
- **Git** (opcional, para clonar repositórios se necessário).
- Uma conta no Hugging Face (gratuita) para acessar modelos: https://huggingface.co/join. Configure um token de acesso (HF_TOKEN) para downloads mais rápidos e sem limites.

## Passo 1: Criar e Configurar o Ambiente Virtual

1. Abra o Prompt de Comando ou PowerShell como administrador.
2. Navegue até o diretório onde deseja criar o projeto (ex.: `cd C:\Users\SeuUsuario\Documents\Projetos`).
3. Crie uma nova pasta para o projeto: `mkdir ProjetoQwenCUDA`.
4. Entre na pasta: `cd ProjetoQwenCUDA`.
5. Crie um ambiente virtual com Python 3.12: `python -m venv .venv312` (o nome ".venv312" indica Python 3.12).
6. Ative o ambiente virtual: `.venv312\Scripts\activate`. Você verá `(.venv312)` no prompt.
7. Atualize o pip para a versão mais recente: `python -m pip install --upgrade pip`.
8. Com o python 3.12 já instalado configure o ambiente: `py -3.12 -m venv .venv312`. Caso ainda não tenha instalado a versão 3.12 então o instale como já indicado na sessão ## Pré-requisitos.

## Passo 2: Instalar Dependências com Suporte a CUDA

1. Crie um arquivo `requirements.txt` na raiz do projeto com o seguinte conteúdo (versões específicas para evitar incompatibilidades):
   ```
   transformers>=5.5.3
   accelerate
   safetensors
   ```
   - Salve o arquivo.

2. Instale os pacotes torch via pip, especificando o índice PyTorch para CUDA 12.1 (ajuste se sua CUDA for diferente):
   ```
   pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
   ```
   - Isso instala PyTorch com suporte a CUDA (torch 2.5.1+cu121 ou superior).

3. Instale as outras dependências especificadas no arquivo requirements.txt:
   ```
   pip install -r requirements.txt
   ```
   - Isso instala transformers (compatível) e accelerate para otimização de GPU.

4. Verifique as instalações:
   - `pip show torch` (deve mostrar versão = 2.5.1 com +cu121).
   - `pip show transformers` (deve ser = 5.5.3).
   - Teste CUDA: Execute `python -c "import torch; print(torch.cuda.is_available())"`. Deve retornar `True` se CUDA estiver funcionando.

## Passo 3: Baixar e Configurar o Modelo

1. Crie uma pasta para cache do modelo: `mkdir Models\Coders\Qwen`.
2. O modelo Qwen2.5-Coder-7B-Instruct será baixado automaticamente na primeira execução, mas você pode pré-baixá-lo para acelerar:
   - Execute temporariamente: `python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-Coder-7B-Instruct', cache_dir='Models/Coders/Qwen')"` (pode levar tempo na primeira vez).
   - A versão quantizada do modelo `Qwen2.5-Coder (Qwen/Qwen2.5-Coder-7B-Instruct-AWQ)` não é suportada nesta versão do Torch. Na verdade, ele precisa da dependência `gptqmodel` instalada que não é compatível com o `torch 2.5.1`. Portanto, modelos AWQ (menores em tamanho) não serão possíveis nesta versão do projeto.  

## Passo 4: Criar o Script Principal

1. Crie um arquivo `main.py` na raiz do projeto. Basei-se no Test.py presente na raiz deste projeto:
2. Salve o arquivo.

## Passo 5: Executar e Testar o Projeto

1. Certifique-se de que o ambiente virtual está ativado: `.venv312\Scripts\activate`.
2. Execute o script: `python main.py`.
   - Se o modelo ainda não foi baixado, então na primeira execução ele será baixado (pode levar 10-30 minutos dependendo da internet/GPU).
   - Você verá progresso de download e carregamento.
   - Warnings sobre HF_TOKEN ou dispositivo são normais e não impedem a execução.
3. Verifique a saída: Deve imprimir o prompt e o texto gerado pelo modelo.
4. Se houver erros:
   - Verifique CUDA: `python -c "import torch; print(torch.cuda.get_device_name(0))"`.
   - Reinstale pacotes se necessário.
   - Para modelos grandes, certifique-se de ter pelo menos 16GB RAM GPU. Nos testes deste guia a NVIDIA RTX 500 Ada Generation possui apenas 4GB VRAM para a GPU. Portanto, extremamente lento.

## Dicas Adicionais e Troubleshooting

### Desempenho
Use `device_map="auto"` para distribuir o modelo entre GPU/CPU. Para GPUs pequenas, considere quantização (ex.: AWQ) ou modelos menores porém outras versões do torch e transformers serão necessárias.

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

### Expansão
Adicione mais funcionalidades, como:
- APIs Flask para servir o modelo
- Scripts para diferentes prompts
- Integração com bancos de dados

### Limpeza
Para desativar o venv, digite `deactivate`.

---

## Resumo da Estrutura do Projeto

```
ProjetoQwenCUDA/
├── .venv312/               # Ambiente virtual
├── Models/
│   └── Coders/
│       └── Qwen/           # Cache dos modelos
├── requirements.txt        # Dependências do projeto
├── main.py                 # Script principal
└── README.md               # Este arquivo
```

# Testes

## Setup
i7 ultra
64 GB DDR5
SSD NVMe 1TB
NVIDIA RTX 500 Ada Generation Laptop GPU 4GB VRAM GDDR6

## Todos os testes usam o mesmo prompt
**Prompt:** Create a minimal api in C# dotnet 10 with an endpoint that returns a JSON response. It should have a single endpoint at /api/hello that returns { "message": "Hello, World!" } but without using any external libraries or controllers.

### Teste 1
**Versões:** torch 2.5.1 e CUDA 12.1

**Total time:** 15:25 min

**Resultado:**

1. Create a new .NET 10 project.
2. Add the necessary code to define the minimal API.

Here's how you can do it:

### Step 1: Create a New .NET 10 Project

Open your terminal and run the following command to create a new .NET 10 Web API project:

```bash
dotnet new web -n MinimalApiExample
cd MinimalApiExample
```

### Step 2: Define the Minimal API

Replace the contents of `Program.cs` with the following code:

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
    return Results.Json(new { message = "Hello, World!" });
});

app.Run();
```

### Explanation

- **WebApplication.CreateBuilder(args)**: This creates a new instance of `WebApplicationBuilder`, which is used to configure the application.

- **builder.Services.AddEndpointsApiExplorer()**: Adds support for generating Swagger documentation automatically.     

- **builder.Services.AddSwaggerGen()**: Adds support for generating Swagger UI.

- **app.MapGet("/api/hello", () => ...)**: Defines a GET endpoint at `/api/hello`. When this endpoint is accessed, it returns a JSON object `{ "message": "Hello, World!" }`.

- **app.Run()**: Starts the Kestrel server and begins listening for requests.

### Running the Application

After saving the changes, you can run the application using the following command:

```bash
dotnet run
```

Once the application is running, open a web browser and navigate to `https://localhost:5001/swagger` (or `http://localhost:5000/swagger` if you're using HTTP). You should see the Swagger UI interface where you can test the `/api/hello` endpoint.

When you click on the `/api/hello` endpoint and execute it, you should see the JSON response:

```json
{
  "message": "Hello, World!"
}
```

This demonstrates a simple minimal API setup
Elapsed time: 925776.46 ms