# Text Emotionalisation

### Abstract

While emotion generation in dialogues for empathetic systems has received special attention from the research community, emotion transfer in texts is a relatively new task. This thesis aims to explore the methods to emotionalise the texts while ensuring fluency and preserving the original semantic meaning. Instead of using unsupervised methods, together with a data-driven approach to the problem of \emph{``style''} and \emph{``content''} as it is normally pursued in literature, we attempt to differentiate the two terms. Our effort, thus, leads to a parallel neutral-emotional corpus. Two Transformer-based sequence-to-sequence architectures are adopted for the implementation of our text emotionalisation models. An additional emotion style loss is employed to guide the generation towards more emotional words and phrases. Before fine-tuning the pre-trained sequence-to-sequence models into the emotionalisation models, we first train them on paraphrase data to refine their re-writing capacity and thus improve the preservation of original content in the generated candidates.
The encouraging results of our initial experiments suggest the potential of our approach. 
Despite having a small-scale corpus, the models are able to emotionalise the input text. The ablation studies are further conducted to understand the contribution of two architecture designs, namely the emotion style loss during training and the pre-training paraphrasing stage.
However, both automatic and human results show that their contribution is modest and unclear. We believe a more comprehensive evaluation is needed to investigate this issue further. 

_A Master Thesis project done at Stuttgart Technology Center, Sony Europe B.V._
