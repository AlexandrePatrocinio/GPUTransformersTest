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

**Versões:** Python 3.12.10, torch 2.5.1, CUDA 12.1, transformers 5.5.3

**LLM:** Qwen/Qwen2.5-Coder-7B-Instruct

**Tamanho total:** 15.23 GB

### Teste 1

**Usando CUDA:** Sim

**Total time:** 14:18 min

### Teste 2

**Usando CUDA:** Não

**Total time:** 38:37 min

### Exemplo de resultado

Creating a minimal API in .NET 10 involves using the `WebApplication` class and defining routes directly within the `Program.cs` file. Below is an example of how you can create a minimal API with an endpoint `/api/hello` that returns a JSON response `{ "message": "Hello, World!" }`.

First, ensure you have the necessary packages installed. If not, you can install them via the NuGet Package Manager:

```sh
dotnet add package Microsoft.AspNetCore.Builder
dotnet add package Microsoft.Extensions.DependencyInjection
dotnet add package Microsoft.Extensions.Hosting
```

Now, create your `Program.cs` file with the following content:

```csharp
using Microsoft.AspNetCore.Builder;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
builder.Services.AddControllers();

var app = builder.Build();

// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
    app.UseDeveloperExceptionPage();
}

app.MapGet("/api/hello", () =>
{
    return new { message = "Hello, World!" };
});

app.Run();
```

### Explanation:
1. **WebApplication.CreateBuilder(args)**: This creates a new instance of `WebApplication` which is used to configure the application.
2. **AddControllers()**: This method adds MVC services to the specified `IServiceCollection`. However, since we're creating a minimal API, this line is optional if you don't need additional MVC features.
3. **MapGet("/api/hello", ...)**: This defines a GET route at `/api/hello`. The lambda function inside `MapGet` returns an anonymous object containing the message property.
4. **Run()**: This starts the web host.

### Running the Application:
To run the application, navigate to the directory containing your `Program.cs` file and execute the following command:

```sh
dotnet run
```

Once the application is running, you can access the endpoint at `http://localhost:5000/api/hello` (or `https://localhost:5001/api/hello` if you're using HTTPS) to see the JSON response:

```json
{ "message": "Hello, World!" }
```

This setup demonstrates how to create a minimal API in .NET 10 with a single endpoint that returns a JSON response.
Elapsed time: 858880.95 ms