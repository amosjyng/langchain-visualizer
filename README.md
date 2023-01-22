# LangChain Visualizer

Adapts [ICE](https://github.com/oughtinc/ice) for use with [LangChain](https://github.com/hwchase17/langchain) so that you can view LangChain interactions with a beautiful UI.

![Screenshot of an execution run](screenshots/serp_screenshot.png "SERP agent demonstration")

You can now

- See the full prompt text being sent with every interaction with the LLM
- Tell from the coloring which parts of the prompt are hardcoded and which parts are templated substitutions
- Inspect the execution flow and observe when each function goes up the stack

### Running the example screenshot

To run the example you see in the screenshot, first install this library and optional dependencies:

```bash
pip install langchain-visualizer google-search-results openai
```

If you haven't yet [set up your OpenAI API keys](https://openai.com/api/), do so now. Then run the following script (adapted from [LangChain docs](https://langchain.readthedocs.io/en/latest/modules/agents/getting_started.html)):

```python
import langchain_visualizer
import asyncio
from langchain.agents import initialize_agent, load_tools
from langchain.llms import OpenAI

llm = OpenAI(temperature=0.7)
tools = load_tools(["serpapi", "llm-math"], llm=llm)
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
async def search_agent_demo():
    return agent.run(
        "Who is Olivia Wilde's boyfriend? What is his current age raised to the 0.23 "
        "power?"
    )

langchain_visualizer.visualize(search_agent_demo)
```

A browser window will open up, and you can actually see the agent execute happen in real-time!

**Please note that there is a lot of langchain functionality that I haven't gotten around to hijacking for visualization.** If there's anything you need to show up in the execution trace, please open a PR or issue.

## My other projects

Please check out [VCR LangChain](https://github.com/amosjyng/vcr-langchain), a library that lets you record LLM interactions for your tests and demos!
