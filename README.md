# Enhanced Tool Calling Framework

إطار عمل مطور لتحسين استدعاء الأدوات بواسطة نماذج اللغة الكبيرة (LLMs)، مع التركيز على النماذج الأقل قوة.

## نظرة عامة

يهدف هذا الإطار إلى حل مشكلة ضعف دقة استدعاء الأدوات في النماذج الأقل قوة من خلال توفير طبقات ذكية للمساعدة في:
- اختيار الأداة المناسبة
- استخراج المعاملات من النص الطبيعي
- تنفيذ الأدوات بشكل موثوق
- التعلم والتحسن من التغذية الراجعة

## الميزات الرئيسية

### 🎯 اختيار ذكي للأدوات
- مطابقة دلالية متقدمة (الإصدار المتقدم)
- مطابقة كلمات مفتاحية محسنة (الإصدار المبسط)
- نظام ثقة وبدائل

### 🔧 استخراج معاملات متطور
- استراتيجيات متعددة للاستخراج
- دعم أنواع بيانات متنوعة
- تحقق تلقائي من الصحة

### ⚡ تنفيذ موثوق
- إعادة محاولة ذكية
- تنفيذ متوازي
- معالجة شاملة للأخطاء

### 📊 تعلم تكيفي
- نظام تغذية راجعة شامل
- مقاييس أداء مفصلة
- تحسين مستمر

## التثبيت السريع

```bash
git clone https://github.com/your-repo/enhanced-tool-calling-framework.git
cd enhanced-tool-calling-framework
```

### المتطلبات الأساسية
- Python 3.11+
- لا توجد مكتبات خارجية مطلوبة للإصدار المبسط

### المتطلبات المتقدمة (اختيارية)
```bash
pip install sentence-transformers spacy
python -m spacy download en_core_web_sm
```

## الاستخدام السريع

### الإصدار المبسط (بدون مكتبات خارجية)

```python
from core.simple_unified_api import SimpleUnifiedToolAPI, ToolCallRequest
from examples.sample_tools import CalculatorTool, WeatherTool

# إنشاء API
api = SimpleUnifiedToolAPI()

# تسجيل الأدوات
api.register_tools([
    CalculatorTool(),
    WeatherTool()
])

# استدعاء أداة
import asyncio

async def main():
    request = ToolCallRequest(
        user_query="what's the weather in London",
        context={}
    )
    
    response = await api.call_tool(request)
    
    if response.success:
        print(f"Tool: {response.tool_used}")
        print(f"Result: {response.result}")
    else:
        print(f"Error: {response.error}")

asyncio.run(main())
```

### الإصدار المتقدم (مع مكتبات خارجية)

```python
from core.unified_api import UnifiedToolAPI, ToolCallRequest
from examples.sample_tools import CalculatorTool, WeatherTool

# إنشاء API مع نماذج متقدمة
api = UnifiedToolAPI(
    embedding_model="all-MiniLM-L6-v2",
    nlp_model="en_core_web_sm"
)

# باقي الكود مشابه للإصدار المبسط
```

## إنشاء أداة مخصصة

```python
from core.base_tool import BaseTool, ToolParameter, ParameterType, ToolResult

class MyCustomTool(BaseTool):
    def __init__(self):
        parameters = [
            ToolParameter(
                name="input_text",
                type=ParameterType.STRING,
                description="النص المراد معالجته",
                required=True
            ),
            ToolParameter(
                name="mode",
                type=ParameterType.STRING,
                description="نمط المعالجة",
                required=False,
                default="basic",
                enum_values=["basic", "advanced"]
            )
        ]
        
        super().__init__(
            name="my_tool",
            description="أداة مخصصة لمعالجة النصوص",
            parameters=parameters
        )
    
    async def execute(self, arguments):
        try:
            text = arguments["input_text"]
            mode = arguments.get("mode", "basic")
            
            # منطق المعالجة هنا
            result = f"Processed '{text}' in {mode} mode"
            
            return ToolResult(
                success=True,
                data=result
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e)
            )
```

## تشغيل الاختبارات

### الاختبار المبسط
```bash
python simple_test_framework.py
```

### الاختبار المتقدم (يتطلب مكتبات خارجية)
```bash
python test_framework.py
```

### تصحيح اختيار الأدوات
```bash
python debug_tool_selection.py
```

## بنية المشروع

```
enhanced_tool_calling_framework/
├── core/                          # المكونات الأساسية
│   ├── base_tool.py              # نظام الأداة الأساسي
│   ├── simple_tool_selector.py   # مختار الأداة المبسط
│   ├── simple_argument_generator.py # مولد المعاملات المبسط
│   ├── simple_unified_api.py     # واجهة برمجة التطبيقات المبسطة
│   ├── execution_layer.py        # طبقة التنفيذ
│   ├── feedback_loop.py          # حلقة التغذية الراجعة
│   └── ...                       # مكونات متقدمة أخرى
├── examples/                     # أمثلة وأدوات للاختبار
│   └── sample_tools.py           # أدوات عينة
├── simple_test_framework.py      # اختبارات مبسطة
└── README.md                     # هذا الملف
```

## الأدوات المتاحة للاختبار

### 🧮 CalculatorTool
- العمليات: جمع، طرح، ضرب، قسمة، أس، جذر تربيعي
- مثال: `"add 15 and 25"`, `"square root of 64"`

### 🌤️ WeatherTool
- بيانات طقس محاكاة للمواقع
- مثال: `"weather in London"`, `"temperature in Paris fahrenheit"`

### 📝 TextAnalyzerTool
- تحليل النصوص وإحصائيات
- مثال: `"analyze this text: Hello world"`, `"sentiment analysis"`

### ⏰ TimeTool
- عمليات الوقت والتاريخ
- مثال: `"what time is it"`, `"current date"`

### 📁 FileInfoTool
- معلومات الملفات (محاكاة)
- مثال: `"info about document.pdf"`

## التكوين المتقدم

### إعدادات التنفيذ

```python
from core.execution_layer import ExecutionConfig, RetryStrategy

config = ExecutionConfig(
    timeout_seconds=30.0,
    max_retries=3,
    retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
    retry_delay=1.0,
    enable_logging=True,
    parallel_execution=True
)

response = await api.call_tool(request, execution_config=config)
```

### مراقبة الأداء

```python
# الحصول على مقاييس الأداء
metrics = api.get_performance_metrics()
print(f"Tool selection accuracy: {metrics['tool_selection_accuracy']}")
print(f"Execution success rate: {metrics['execution_success_rate']}")

# الحصول على اقتراحات التحسين
suggestions = api.get_improvement_suggestions()
for suggestion in suggestions:
    print(f"Component: {suggestion['component']}")
    print(f"Issue: {suggestion['issue']}")
    print(f"Suggestion: {suggestion['suggestion']}")
```

## المساهمة

نرحب بالمساهمات! يرجى:

1. فتح issue لمناقشة التغييرات المقترحة
2. إنشاء fork للمشروع
3. إنشاء branch للميزة الجديدة
4. إضافة اختبارات للكود الجديد
5. إرسال pull request

## الترخيص

هذا المشروع مرخص تحت رخصة MIT - راجع ملف [LICENSE](LICENSE) للتفاصيل.

## الدعم

- 📧 البريد الإلكتروني: support@example.com
- 💬 المناقشات: [GitHub Discussions](https://github.com/your-repo/discussions)
- 🐛 الأخطاء: [GitHub Issues](https://github.com/your-repo/issues)

## الشكر والتقدير

- مستوحى من Google ADK Python
- يستخدم مكتبات مفتوحة المصدر رائعة
- شكر خاص للمجتمع المطور

---

**ملاحظة**: هذا المشروع في مرحلة التطوير النشط. نرحب بالتغذية الراجعة والاقتراحات!



## دمج نماذج اللغة الكبيرة (LLM)

تم تصميم هذا الإطار ليكون مرنًا، مما يسمح لك بدمج أي نموذج لغة كبير (LLM) تختاره، سواء كان نموذجًا قائمًا على السحابة مثل Google Gemini أو نموذجًا محليًا. يركز الإطار على تعزيز قدرة LLM على استدعاء الأدوات بذكاء من خلال توفير طبقات مساعدة متخصصة.

### استخدام LlmAgent مع Gemini

يمكنك استخدام `LlmAgent` كوكيل رئيسي يتفاعل مع نموذج Gemini (أو أي LLM آخر) ويستخدم الأدوات التي تحددها. يتم قراءة مفتاح Google AI API تلقائيًا من ملف `.env` الخاص بك.

**1. إعداد ملف `.env`:**

قم بإنشاء ملف باسم `.env` في جذر مشروعك (نفس مستوى `README.md`) وأضف مفتاح API الخاص بك:

```
GOOGLE_AI_API_KEY=YOUR_GEMINI_API_KEY_HERE
```

**2. تعريف الأدوات:**

يمكنك تعريف أدواتك كوظائف Python عادية، ثم تمريرها إلى `LlmAgent`. سيقوم الإطار تلقائيًا بتحويلها إلى تنسيق يمكن لـ LLM فهمه.

```python
def get_weather(cityName: str):
    """
    return the current weather in a specific city
    Args:
        cityName(str): the city name
    Return:
        the current weather of a specific city 
    """
    return f"The current weather in {cityName} is 35 degrees Celsius and sunny."
```

**3. تهيئة LlmAgent وتشغيله:**

```python
import asyncio
import os
from enhanced_tool_calling_framework.llm_integration.root_agent import LlmAgent, load_env_file
from enhanced_tool_calling_framework.llm_integration.gemini_wrapper import GeminiWrapper

# تحميل متغيرات البيئة من ملف .env
load_env_file()
api_key = os.getenv("GOOGLE_AI_API_KEY")

if not api_key:
    print("Error: GOOGLE_AI_API_KEY not found in .env file. Please provide it.")
else:
    # تهيئة نموذج Gemini LLM
    model = GeminiWrapper(model_name="gemini-2.0-flash", api_key=api_key)

    # تهيئة الوكيل الرئيسي
    root_agent = LlmAgent(
        model=model,
        name=\'weather_provider\',
        instruction=(
            "أنت مساعد طقس مفيد. استخدم الأدوات المتاحة للإجابة على أسئلة المستخدم حول الطقس."
        ),
        tools=[get_weather],
        language="ar"
    )

    async def run_agent_example():
        print(f"Agent initialized: {root_agent}")
        print(f"Available tools: {root_agent.list_tools()}")

        # اختبار الاستعلامات
        test_queries = [
            "ما هو الطقس في صفاقس، تونس؟",
            "هل يمكنك أن تخبرني عن الطقس في لندن؟",
            "مرحباً، كيف حالك؟"
        ]

        for query in test_queries:
            print(f"\nUser Query: {query}")
            print("-" * 30)
            response = await root_agent.run_async(query)
            print(f"Agent Response: {response.message}")
            if response.tool_used:
                print(f"  Tool Used: {response.tool_used}")
                print(f"  Tool Result: {response.tool_result}")
            print(f"  Success: {response.success}")
            print(f"  Reasoning: {response.reasoning}")
            if response.error:
                print(f"  Error: {response.error}")

    if __name__ == "__main__":
        asyncio.run(run_agent_example())
```

بهذا الإعداد، يمكن لـ `LlmAgent` التفاعل مع Gemini، وتزويده بوصف الأدوات، ثم تحليل استجابة Gemini لتحديد الأداة المناسبة واستخراج المعاملات وتنفيذها، مما يوفر تجربة ذكية لاستدعاء الأدوات.

