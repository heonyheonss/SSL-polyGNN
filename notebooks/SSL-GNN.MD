# SSL-GNN

## SSL

contrastive : training data - data -> utilize self-supervision to learn data rep : discr- positive & negative  
predictive : training data - generated label : more than one encoder 
 - label : (1) certain props / (2) selected part

## To obtain good rep of Gs and eff- pre-training
Good model should capture important info(nodes attr, struct- topo-)

cont : to obtain good views of G and selection of graph encoder (models and datasets)
pred : gen only important label to capture (1) node att (2) graph struct

## Tree of SSL methods
### Contrastive
(1) Objective: Jensen-Shannon estimator, InfoNCE/NT-Xent, Other MI estimators
(2) Views: Identical, Subgraphs, Structure TF, Feature TF, Subgraphs → Structure TF, Structure TF/feature TF/subgraph

### Predictive
(1) Graph reconstruction: Non-probabilistic graph auto encoder, Variational graph autoencoder, Autoregressive reconstruction
(2) Property prediction: Statistical, Domain-knowledge, Topological
(3) Self-training
(4) Invariance regularization

