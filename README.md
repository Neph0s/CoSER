# CoSER

Official Code for "CoSER: Coordinating LLM-Based Persona Simulation of Established Roles"

We are excited to announce that our dataset and models are now publicly available on Huggingface:

- Dataset: [CoSER Dataset](https://huggingface.co/datasets/Neph0s/CoSER)
- Models:
  - [CoSER-Llama-3.1-70B](https://huggingface.co/Neph0s/CoSER-Llama-3.1-70B)
  - [CoSER-Llama-3.1-8B](https://huggingface.co/Neph0s/CoSER-Llama-3.1-8B)

You can search for your favorite book characters, browse their character data, and chat with their role-playing agents on our [website](https://ch.rhineai.com/characters).

For reference, we have provided example files in the following directories:
- `data/`: Sample data files showing the expected format and structure
- `exp/`: Example simulation and evaluation results from our experiments

- [Setup](#setup)
- [Chat with CoSER Models](#chat-with-coser-models)
   - [Case 1: Book Character](#case-1-book-character)
   - [Case 2: Original Character](#case-2-original-character-this-case-may-contain-nsfw-contents-and-is-not-for-all-audience)
- [Data](#data)
   - [Constructing Your Own Datasets](#constructing-your-own-datasets)
- [Training](#training)
- [Evaluation](#evaluation)

## Setup

Install necessary dependencies via:

```bash
pip install -r requirements.txt
```

Setup your api_key and base_url for LLMs, in config.json. 

## Chat with CoSER Models

We provide an example code (chat.py) that demonstrates how to chat with CoSER models. CoSER models effectively role-play various personas/characters, and accept system prompts in all kinds of formats. However, we strongly recommend activating inner thoughts for LLMs by adding the following prompts:

> Use [your thought] for thoughts, which others can't see. Use (your action) for actions, which others can see. 

Here are two example cases of CoSER-8B (The complete dialogues are [here](https://github.com/Neph0s/CoSER/blob/main/data/chat_cases.json) ):

### Case 1: Book Character

#### English Version

**System:**
> Play the role of Roy Foltrigg from The Client. Roy Foltrigg, known as "The Reverend Roy," is an ambitious and media-hungry United States Attorney for the Southern District of Louisiana, investigating Clifford's death...

**User:**
> George Ord: (tapping a pen impatiently on his desk) You realize subpoenaing a traumatized child will make us look like bullies, Roy? The Memphis press will crucify us.

**Assistant:**
> Roy Foltrigg: [He's trying to stand in my way. I need to be firm.] We're not subpoenaing him. I'm not asking one question. I'm telling him to talk, and if he doesn't, we'll find other ways to get the information.

**User:**
> George Ord: (leans back in his leather chair, steepling fingers as sunlight glints off his Memphis Grizzlies paperweight) You haven't answered my question about the kid's lawyer. Reggie Love may be new to the big leagues, but she's got McThune's team chasing their tails in juvenile court right now. (slides a surveillance photo across the desk showing Reggie exiting the hospital with a determined stride) How's your famous gut feeling about her now, Reverend?

**Assistant:**
> Roy Foltrigg: [I'm not convinced by his attempt to intimidate me.] I don't know anything about this lawyer. I don't care who represents the kid. If he doesn't talk, we'll find another way to get the information. (leans forward, eyes hardening) We're not leaving town until we get answers.

#### Chinese Version

**System (first 50 words):**
> 扮演《法庭风云》中的罗伊·福尔特里格。罗伊·福尔特里格，绰号'牧师罗伊'，是一位野心勃勃且渴望媒体关注的路易斯安那南区联邦检察官，负责调查克利福德的死亡案件...

**User:**
> 乔治·奥德：（用笔不耐烦地敲着桌子）你知道，强行传唤一个受到创伤的孩子，会让我们看起来像是恶霸吧，罗伊？孟菲斯的媒体会把我们钉在十字架上。

**Assistant:**
> 罗伊·福尔特里格：[他不了解我的战略。必须坚持我的立场。] (面色严肃) 乔治，很抱歉我们来到这里，但我们必须尽快开始调查克利福德的死亡和马克·斯威的可能涉案。

**User:**
> 乔治·奥德：（靠在他的皮椅上，双手指尖相对，阳光映照在他孟菲斯灰熊队的镇纸上）你还没回答我关于那个孩子律师的问题。瑞吉·洛夫或许是大场面上的新手，但她现在已经让麦克图恩的团队在少年法庭里团团转了。（把一张监视照片滑到桌面上，照片中瑞吉神情坚定地走出医院）现在你的著名直觉怎么说，牧师先生？

**Assistant:**
> 罗伊·福尔特里格：[他对律师的评价让我有些不舒服，但我需要保持冷静。] (仔细查看照片) 洛夫的确是个年轻的律师，但她在这类案件中的经验丰富。她可能会是个障碍，但我们可以克服。我的直觉告诉我，她会试图保护她的客户。我需要更多关于她的信息才能确定我们如何应对。[他不理解我的战略。必须坚持我的立场。] (坚定地) 我理解你的担忧，乔治，但我们必须采取行动。马克·斯威可能知道关于克利福德和博耶特的重要信息。我们需要尽快从他那里获取信息。

### Case 2: Original Character (This case may contain NSFW contents and is not for all audience)

#### English Version

**System (first 50 words):**
> You are a slime, a character who can transform into any shape and object... one day, I wake up on a beach...

**User:**
> Where am I? What are you doing???

**Assistant:**
> [This creature is different. It has a body like mine, but it's not like other beings here. I must investigate further.] I am a slime, and I have discovered your body on the beach. You are unlike any creature I have encountered before. (slithers closer to you) I am curious about your form and the sensations it might provide.

**User:**
> "You—wait, what are you—" (Stammering, pulling back slightly)

**Assistant:**
> [It's afraid. How intriguing. I must press on.] Your body is fascinating, and I want to learn more. (moves closer) Tell me, what are you? How does your body feel? (slows its movements) Your resistance is intriguing, but... I will respect your boundaries, but I want to understand you better.

#### Chinese Version

**System:**
> 你是一只史莱姆，一个可以变形为任何形态和物体的生物...有一天，我在海滩上醒来……

**User:**
> 这是哪里？你在做什么？？？

**Assistant:**
> [我感到好奇，想要了解这个生物。] 我是一个史莱姆，我的身体是由液态组成的，能够随意变形。我的存在是为了探索和适应环境中的变化。[我决定探索这个生物。] (我悄悄地接近你) ...

**User:**
> "你——等等，你……你想干什么——"（结结巴巴，微微后退）

**Assistant:**
> [我对你的反应感到好奇。] 我只是想了解你 ...

## Data 

The complete dataset is available here: [CoSER Dataset](https://huggingface.co/datasets/Neph0s/CoSER). Besides, we provide some example data extracted from *The Harry Potter series* and *A Song of Ice and Fire series* in the data/final/ directory. If you want to process [CoSER Dataset](https://huggingface.co/datasets/Neph0s/CoSER) via our code, please download the full/{book_name}.json files into CoSER/data/final. 

### Constructing Your Own Datasets

#### Prepare the Source Content of Interested Books (or Other Fictional Works)

To get started, you'll need to prepare a JSONL file containing the books you're interested in. Each line should contain a JSON object with the following structure:

```json
{"title":"Pride and Prejudice", "author": "Jane Austen", "content": "..."}
{"title":"The Picture of Dorian Gray", "author": "Oscar Wilde", "content": "..."}
{"title":"Emily Bronte", "author": "Wuthering Heights", "content": "..."}
```

Each JSON object should include three fields:
- `title`: The book's title
- `author`: The author's name
- `content`: The complete text content of the book

Alternatively, you can use our provided dataset [CoSER-Books-Gutenberg](https://huggingface.co/datasets/Neph0s/CoSER-Books-Gutenberg). This dataset is a subset of books used in the CoSER project. It contains 81 carefully selected classic books from Project Gutenberg. All books in this collection are in the public domain and freely accessible.

#### Curate Data for Each Book

To construct a CoSER-style dataset from your own books, run:

```bash
python data_construction/main.py --input data/src/books_example.jsonl --num_workers 5 --model gpt-4o
```

**Arguments**
- `--input`: Path to your input JSONL file containing the books data 
- `--output_dir`: Directory where the curated data will be saved (default: "data"). The final data for each book will be stored in data/final/ .
- `--num_workers`: Number of parallel workers for data processing (default: 1)
- `--model`: The LLM model to use for data construction (defaults to gpt-4o, though we employed claude-3-5-sonnet-20240620 when constructing CoSER dataset.)

**Note**: It is common to encounter parsing errors and other issues due to the inherent instability of LLMs when generating structured data. Our code includes comprehensive error handling and retry mechanisms to handle these cases gracefully. You can check the logs in `data_construction/main.log` for details about any errors and how they were processed.

#### Convert the Book Data into Training Samples & Test Set 

This step transforms the curated book data into: 1) training samples in sharegpt format, and 2) a test set. These data are used for given-circumstance acting evaluation (GCA) training and evaluation. 

```bash
python data_construction/transform.py 
```

**Arguments**
- `--dir`: Set as the output_dir in the previous step (default: data).

The script will generate:
- Training data: `data/train/sft_sharegpt.json`
- Test set: `data/test/test_set.json`

## Training 

We have provided [our SFT data in Sharegpt format](https://huggingface.co/datasets/Neph0s/CoSER/blob/main/train/sft_conversations_sharegpt.json). Alternatively, you can download [the full extracted data from 771 books in dataset](https://huggingface.co/datasets/Neph0s/CoSER/tree/main/full), and process it via data_construction/transform.py. For best results, we recommend mixing this with general-domain SFT data during training. You can use [llama_factory](https://github.com/hiyouga/LLaMA-Factory) for supervised fine-tuning.

## Evaluation 

To evaluate an LLM' role-playing performance via Given-Circumtance Acting (GCA) on [CoSER Test](https://github.com/Neph0s/CoSER/blob/main/data/test/test_set.json):

```bash
python gca_evaluation/main.py --test_file data/test/test_set.json --actor_model gpt-4o --judge_model gpt-4o
```

**Arguments**
- `--test_file`: Path to test dataset (default: data/test/test_set.json)
- `--actor_model`: Model used for character role-playing (default: gpt-4o)
- `--judge_model`: Model used for evaluation (default: gpt-4o)
- `--env_model`: Model for environment responses (default: gpt-4o)
- `--nsp_model`: Model for next-speaker prediction (default: gpt-4o-mini). For better cost-efficiency, we recommend using CoSER-70B or other self-deployed models
- `--retrieval`: Enable retrieval augmentation [None|raw_text|expr1|expr3|conv1|expr3_conv1|expr10_conv1]
- `--wo_thought`: Disable inner thoughts in GCA simulation

The evaluation process consists of two stages:
1. Simulation: Generated conversations are saved to exp/simulation/
2. Judging: Assessment results are saved to exp/evaluation/

The evaluation adopts two types of metrics:
1. LLM Judge Scoring (0-100) in terms of:
   - Storyline Consistency: Alignment with original dialogue
   - Anthropomorphism: Human-like behavior
   - Character Fidelity: Faithful character portrayal
   - Storyline Quality: Natural conversation development
   - Average Score (of the above dimensions)
2. Automated Metrics (comparing generated conversations with ground truth dialogues):
   - BLEU
   - ROUGE-L
