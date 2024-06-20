# 深入理解LCEL: 可编排Langchain组件的“灵魂”

- 2024-06-17
- hcd233

### 前言

如今`Langchain`已是大语言模型应用开发的事实标准框架，早在Langchain推出第一个稳定版本v0.1.0时，就已提出了一种新的Langchain范式：`Langchain Expression Language`简称`LCEL`，在本篇文章中将会全面地介绍LCEL的设计思想以及具体的实现。

### 设计思想：一切组件皆可编排

早在去年9月时，我便在鹅厂参与一款AI应用的开发，当时我们有一个功能，需要**初始化三个不同的LLMChain，并且将他们串联到一起**，我们当时是这么做的

```python
chain1 = LLMChain(
    prompt=prompt1,
    llm=llm,
)
chain2 = LLMChain(
    prompt=prompt2,
    llm=llm,
)
chain3 = LLMChain(
    prompt=prompt3,
    llm=llm,
)

overall_chain = SequentialChain(
    chains=[chain1, chain2, chain3],
)
```

这样子便可以将三个chain给串联起来，看起来似乎很简单，但是当我们开始**用起来LCEL后，发现还能更简单**

```python
chain1 = prompt1 | llm
chain2 = prompt2 | llm
chain3 = prompt3 | llm

overall_chain = chain1 | chain2 | chain3
```

我们直接用类似管道符号"`|`"将prompt和llm串联在一起，便可以组成一个LLMChain，将chain和chain串联在一起，便可以组成一个SequentialChain!

值得让人兴奋的是，不仅仅是`PromptTemplate`，`ChatModel`，基本上你见过的Langchain组件，你都可以通过管道符号进行串联，例如`Retriever`，`OutputParser`，`Tool`等等。甚至可以是你写的python函数。

这里其实LCEL的思想便可见一斑了，对于Langchain的大部分组件，其实都有明确的输入输出，常用组件的输入输出如下：

| 组件 | 输入 | 输出 |
|--|--|--|
| LLM | List[Message] | AIMessage/ToolMessage |
| PromptTemplate | Dict[str, Any] | List[Message] |
| OutputParser | Any | Any |
| Retriever | str | List[Document] |
| Tool | Dict[str, Any] | Any |

那么仔细一想，如果说我们能给Langchain的这些组件制定统一的规范，并且重写和实现一些魔术方法，是不是就可以实现组件的自由编排了？

事实上确实如此，接下来，我们来一起看看LCEL的底层是怎样给组件定下统一的规范的

### Runnable：LCEL的驱动核心

如果你使用Langchain来构建AI应用，则或多或少都会接触过Runnable，这里无需多言，直接看源码

#### Runnable：最底层的抽象

首先直接看源码，在包langchain_core.runnable.base中可以清晰看到Runnable的定义

```python
class Runnable(Generic[Input, Output], ABC):
    """A unit of work that can be invoked, batched, streamed, transformed and composed.
```

这里可以看到，Runnable继承了一个具有Input和Output变量的泛型，以及抽象基类(ABC)，那么它就会有如下特征：

1. 首先对于泛型，这里有点类似C++的模板的意味，只不过在动态类型的python中，这里主要起变量注解的作用，这里主要标识任何Runnable以及子类都会表明自己的Input和Output

2. 对于ABC，则说明Runnable是最底层的抽象，会定义一系列抽象方法来规范子类

这里就可以看看Runnable提供了哪些抽象接口

```python
@abstractmethod
    def invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Output:
        """Transform a single input into an output. Override to implement.
        """

def stream(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Iterator[Output]:

def batch(
        self,
        inputs: List[Input],
        config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Optional[Any],
    ) -> List[Output]:

def assign(
        self,
        **kwargs: Union[
            Runnable[Dict[str, Any], Any],
            Callable[[Dict[str, Any]], Any],
            Mapping[
                str,
                Union[Runnable[Dict[str, Any], Any], Callable[[Dict[str, Any]], Any]],
            ],
        ],
    ) -> RunnableSerializable[Any, Any]:

def pick(self, keys: Union[str, List[str]]) -> RunnableSerializable[Any, Any]:

@beta_decorator.beta(message="This API is in beta and may change in the future.")
    async def astream_events(
        self,
        input: Any,
        config: Optional[RunnableConfig] = None,
        *,
        version: Literal["v1"],
        include_names: Optional[Sequence[str]] = None,
        include_types: Optional[Sequence[str]] = None,
        include_tags: Optional[Sequence[str]] = None,
        exclude_names: Optional[Sequence[str]] = None,
        exclude_types: Optional[Sequence[str]] = None,
        exclude_tags: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:

def __or__(
        self,
        other: Union[
            Runnable[Any, Other],
            Callable[[Any], Other],
            Callable[[Iterator[Any]], Iterator[Other]],
            Mapping[str, Union[Runnable[Any, Other], Callable[[Any], Other], Any]],
        ],
    ) -> RunnableSerializable[Input, Other]:
        """Compose this runnable with another object to create a RunnableSequence."""
        return RunnableSequence(self, coerce_to_runnable(other))
```

这些基本上就是`Runnable`最常用的方法了，其中`invoke`是抽象方法，意味着只要继承Runnable类你都必须重写这个`invoke`方法，并且`stream`, `batch`等方法都依赖`invoke`方法的实现，这里还可以看到它实现了__or__方法，通过`RunnableSequence`实现Runnable的组合，这里返回的变量注解写的`RunnableSerializable`类在后面会讲到

接下来我们直接来看他有哪些子类，这里如果你仔细看下来，你会发现只有一个叫`RunnableSerializable`的类继承了它，这里`RunnableSerializable`继承了Langchain的一个工具基类`Serializable`，用于方便Langchain组件进行序列化。

#### RunnableSerializable：组件的统一接口

```python
class RunnableSerializable(Serializable, Runnable[Input, Output]):
    """Runnable that can be serialized to JSON."""

class Serializable(BaseModel, ABC):
    """Serializable base class."""
```

这里我们可以确定，`RunnableSerializable`才是Langchain组件统一继承的接口，`Runnable`相当于只是为组件提供基本抽象的接口

#### RunnableSequence：串联Runnable的子类

这个`RunnableSequence`有点`SequentialChain`的感觉了，这里我放关键的源码出来，挺好看懂的

```python
class RunnableSequence(RunnableSerializable[Input, Output]):
    """Sequence of Runnables, where the output of each is the input of the next."""
    first: Runnable[Input, Any]
    """The first runnable in the sequence."""
    middle: List[Runnable[Any, Any]] = Field(default_factory=list)
    """The middle runnables in the sequence."""
    last: Runnable[Any, Output]
    """The last runnable in the sequence."""

    @property
    def steps(self) -> List[Runnable[Any, Any]]:
        """All the runnables that make up the sequence in order."""
        return [self.first] + self.middle + [self.last]

    def __or__(
        self,
        other: Union[
            Runnable[Any, Other],
            Callable[[Any], Other],
            Callable[[Iterator[Any]], Iterator[Other]],
            Mapping[str, Union[Runnable[Any, Other], Callable[[Any], Other], Any]],
        ],
    ) -> RunnableSerializable[Input, Other]:
        if isinstance(other, RunnableSequence):
            return RunnableSequence(
                self.first,
                *self.middle,
                self.last,
                other.first,
                *other.middle,
                other.last,
                name=self.name or other.name,
            )
        else:
            return RunnableSequence(
                self.first,
                *self.middle,
                self.last,
                coerce_to_runnable(other),
                name=self.name,
            )


    def invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Output:
        from langchain_core.beta.runnables.context import config_with_context

        # setup callbacks and context
        config = config_with_context(ensure_config(config), self.steps)
        callback_manager = get_callback_manager_for_config(config)
        # start the root run
        run_manager = callback_manager.on_chain_start(
            dumpd(self),
            input,
            name=config.get("run_name") or self.get_name(),
            run_id=config.pop("run_id", None),
        )

        # invoke all steps in sequence
        try:
            for i, step in enumerate(self.steps):
                input = step.invoke(
                    input,
                    # mark each step as a child run
                    patch_config(
                        config, callbacks=run_manager.get_child(f"seq:step:{i+1}")
                    ),
                )
        # finish the root run
        except BaseException as e:
            run_manager.on_chain_error(e)
            raise
        else:
            run_manager.on_chain_end(input)
            return cast(Output, input)

```

看到这里应该有点感觉了，这里`RunnableSequence`就是为了串联Runnable而存在的，还记得我们前面的那段代码实例吗？在执行完overall_chain = chain1 | chain2 | chain3后，这里的overall_chain就会是一个`RunnableSequence`对象

#### RunnableParallel：并行Runnable的子类

这个`RunnableParallel`我比较少用，这里还是直接贴关键源码，大致明白它的原理就ok了

```python
class RunnableParallel(RunnableSerializable[Input, Dict[str, Any]]):
    steps__: Mapping[str, Runnable[Input, Any]]

    def invoke(
        self, input: Input, config: Optional[RunnableConfig] = None
    ) -> Dict[str, Any]:
        from langchain_core.callbacks.manager import CallbackManager

        # setup callbacks
        config = ensure_config(config)
        callback_manager = CallbackManager.configure(
            inheritable_callbacks=config.get("callbacks"),
            local_callbacks=None,
            verbose=False,
            inheritable_tags=config.get("tags"),
            local_tags=None,
            inheritable_metadata=config.get("metadata"),
            local_metadata=None,
        )
        # start the root run
        run_manager = callback_manager.on_chain_start(
            dumpd(self),
            input,
            name=config.get("run_name") or self.get_name(),
            run_id=config.pop("run_id", None),
        )

        # gather results from all steps
        try:
            # copy to avoid issues from the caller mutating the steps during invoke()
            steps = dict(self.steps__)
            with get_executor_for_config(config) as executor:
                futures = [
                    executor.submit(
                        step.invoke,
                        input,
                        # mark each step as a child run
                        patch_config(
                            config,
                            callbacks=run_manager.get_child(f"map:key:{key}"),
                        ),
                    )
                    for key, step in steps.items()
                ]
                output = {key: future.result() for key, future in zip(steps, futures)}
        # finish the root run
        except BaseException as e:
            run_manager.on_chain_error(e)
            raise
        else:
            run_manager.on_chain_end(output)
            return output

```

这里看下来`RunnableParallel`只是开了个线程池来invoke map中所有的runnable对象，没啥意思。


#### RunnableLambda: 将函数封装成Runnable

当时LCEL还有一个特性，就是可以用管道`"|"`直接连接函数和Runnable，实际上底层就将函数转换成了RunnableLambda，这里也放出关键的源码

```python
class RunnableLambda(Runnable[Input, Output]):
    """RunnableLambda converts a python callable into a Runnable."""
    def invoke(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Output:
        """Invoke this runnable synchronously."""
        if hasattr(self, "func"):
            return self._call_with_config(
                self._invoke,
                input,
                self._config(config, self.func),
                **kwargs,
            )
        else:
            raise TypeError(
                "Cannot invoke a coroutine function synchronously."
                "Use `ainvoke` instead."
            )

    def _invoke(
        self,
        input: Input,
        run_manager: CallbackManagerForChainRun,
        config: RunnableConfig,
        **kwargs: Any,
    ) -> Output:
        if inspect.isgeneratorfunction(self.func):
            output: Optional[Output] = None
            for chunk in call_func_with_variable_args(
                cast(Callable[[Input], Iterator[Output]], self.func),
                input,
                config,
                run_manager,
                **kwargs,
            ):
                if output is None:
                    output = chunk
                else:
                    try:
                        output = output + chunk  # type: ignore[operator]
                    except TypeError:
                        output = chunk
        else:
            output = call_func_with_variable_args(
                self.func, input, config, run_manager, **kwargs
            )
        # If the output is a runnable, invoke it
        if isinstance(output, Runnable):
            recursion_limit = config["recursion_limit"]
            if recursion_limit <= 0:
                raise RecursionError(
                    f"Recursion limit reached when invoking {self} with input {input}."
                )
            output = output.invoke(
                input,
                patch_config(
                    config,
                    callbacks=run_manager.get_child(),
                    recursion_limit=recursion_limit - 1,
                ),
            )
        return cast(Output, output)
```

这里我理解`RunnableLambda`只不够是func的一层wrapper，实际上只是为了把函数转换成RunnableLambda而已


