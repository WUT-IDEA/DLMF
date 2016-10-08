#include "common.hpp"

class topicCorpus
{
public:
  topicCorpus(corpus* corp, // The corpus
              int K, // The number of latent factors
              double latentReg, // Parameter regularizer used by the "standard" recommender system
              double lambda1,
			  double lambda2) : // Word regularizer used by HFT
    corp(corp), K(K), latentReg(latentReg), lambda1(lambda1),lambda2(lambda2)
  {
    srand(0);

    nUsers = corp->nUsers;
    nBeers = corp->nBeers;
    nWords = corp->nWords;

    votesPerUser = new std::vector<vote*>[nUsers];
    votesPerBeer = new std::vector<vote*>[nBeers];
    trainVotesPerUser = new std::vector<vote*>[nUsers];
    trainVotesPerBeer = new std::vector<vote*>[nBeers];

    for (std::vector<vote*>::iterator it = corp->V->begin(); it != corp->V->end(); it++)
    {
      vote* vi = *it;
      votesPerUser[vi->user].push_back(vi);
    }

    for (int user = 0; user < nUsers; user++)
      for (std::vector<vote*>::iterator it = votesPerUser[user].begin(); it != votesPerUser[user].end(); it++)
      {
        vote* vi = *it;
        votesPerBeer[vi->item].push_back(vi);
      }

    double testFraction = 0.1;
    if (corp->V->size() > 2400000)
    {
      double trainFraction = 2000000.0 / corp->V->size();
      testFraction = (1.0 - trainFraction)/2;
    }

    for (std::vector<vote*>::iterator it = corp->V->begin(); it != corp->V->end(); it ++)
    {
      double r = rand() * 1.0 / RAND_MAX;
      if (r < testFraction)
      {
        testVotes.insert(*it);
      }
      else if (r < 2*testFraction)
        validVotes.push_back(*it);
      else
      {
        trainVotes.push_back(*it);
        trainVotesPerUser[(*it)->user].push_back(*it);
        trainVotesPerBeer[(*it)->item].push_back(*it);
        if (nTrainingPerUser.find((*it)->user) == nTrainingPerUser.end())
          nTrainingPerUser[(*it)->user] = 0;
        if (nTrainingPerBeer.find((*it)->item) == nTrainingPerBeer.end())
          nTrainingPerBeer[(*it)->item] = 0;
        nTrainingPerUser[(*it)->user] ++;
        nTrainingPerBeer[(*it)->item] ++;
      }
    }

    std::vector<vote*> remove;
    for (std::set<vote*>::iterator it = testVotes.begin(); it != testVotes.end(); it ++)
    {
      if (nTrainingPerUser.find((*it)->user) == nTrainingPerUser.end()) remove.push_back(*it);
      else if (nTrainingPerBeer.find((*it)->item) == nTrainingPerBeer.end()) remove.push_back(*it);
    }
    for (std::vector<vote*>::iterator it = remove.begin(); it != remove.end(); it ++)
    {
      // Uncomment the line below to ignore (at testing time) users/items that don't appear in the training set
//      testVotes.erase(*it);
    }

    // total number of parameters
//    NW = 1 + 1 + (K + 1) * (nUsers + nBeers) + K * nWords;
    NW = 1 + 2 + (K + 1) * (nUsers + nBeers) + 2*K * nWords;


    // Initialize parameters and latent variables
    // Zero all weights
    W = new double [NW];
    for (int i = 0; i < NW; i++)
      W[i] = 0;
    getG(W, &alpha, &kappa1,&kappa2,&beta_user, &beta_beer, &gamma_user, &gamma_beer, &topicWords1, &topicWords2, true);

    // Set alpha to the average
    for (std::vector<vote*>::iterator vi = trainVotes.begin(); vi != trainVotes.end(); vi++)
    {
      *alpha += (*vi)->value;
    }
    *alpha /= trainVotes.size();

    double train, valid, test, testSte;
    validTestError(train, valid, test, testSte);
    printf("Error w/ offset term only (train/valid/test) = %f/%f/%f (%f)\n", train, valid, test, testSte);

    // Set beta to user and product offsets
    for (std::vector<vote*>::iterator vi = trainVotes.begin(); vi != trainVotes.end(); vi++)
    {
      vote* v = *vi;
      beta_user[v->user] += v->value - *alpha;
      beta_beer[v->item] += v->value - *alpha;
    }
    for (int u = 0; u < nUsers; u++)
      beta_user[u] /= votesPerUser[u].size();
    for (int b = 0; b < nBeers; b++)
      beta_beer[b] /= votesPerBeer[b].size();
    validTestError(train, valid, test, testSte);
    printf("Error w/ offset and bias (train/valid/test) = %f/%f/%f (%f)\n", train, valid, test, testSte);

    // Actually the model works better if we initialize none of these terms
    if (lambda1 > 0||lambda2>0)
    {
      *alpha = 0;
      for (int u = 0; u < nUsers; u++)
        beta_user[u] = 0;
      for (int b = 0; b < nBeers; b++)
        beta_beer[b] = 0;
    }

      wordTopicCounts1 = new int*[nWords];
      wordTopicCounts2 = new int*[nWords];
      for (int w = 0; w < nWords; w++)
      {
        wordTopicCounts1[w] = new int[K];
        wordTopicCounts2[w] = new int[K];
        for (int k = 0; k < K; k++){
        	 wordTopicCounts1[w][k] = 0;
        	 wordTopicCounts2[w][k] = 0;
        }

      }

    // Generate random topic assignments
    topicCounts1 = new long long[K];
    topicCounts2 = new long long[K];
    for (int k = 0; k < K; k++){
    	 topicCounts1[k] = 0;
    	 topicCounts2[k] = 0;
    }

    beerTopicCounts = new int*[nBeers];
    userTopicCounts = new int*[nUsers];

    beerWords = new int[nBeers];
    userWords = new int[nUsers];
    for (int b = 0; b < nBeers; b ++)
    {
      beerTopicCounts[b] = new int[K];
      for (int k = 0; k < K; k ++)
        beerTopicCounts[b][k] = 0;
      beerWords[b] = 0;
    }

    //³õÊ¼»¯userTopicCountsºÍuserWords
        for (int u = 0; u < nUsers; u ++)
        {
    	  userTopicCounts[u] = new int[K];
    	  for (int k = 0; k < K; k ++)
    		 userTopicCounts[u][k] = 0;
    	  userWords[u] = 0;
        }

    for (std::vector<vote*>::iterator vi = trainVotes.begin(); vi != trainVotes.end(); vi++)
    {
      vote* v = *vi;
      wordTopics1[v] = new int[v->words.size()];
           wordTopics2[v] = new int[v->words.size()];

           beerWords[(*vi)->item] += v->words.size();
           userWords[(*vi)->user] +=v->words.size();


      for (int wp = 0; wp < (int) v->words.size(); wp++)
      {
        int wi = v->words[wp];
        int t = rand() % K;

        wordTopics1[v][wp] = t;
        beerTopicCounts[(*vi)->item][t]++;
        wordTopicCounts1[wi][t]++;
        topicCounts1[t]++;


        t = rand() % K;

            wordTopics2[v][wp] = t;
               userTopicCounts[(*vi)->user][t]++;
               wordTopicCounts2[wi][t]++;
               topicCounts2[t]++;
      }
    }

    // Initialize the background word frequency
    totalWords = 0;
    backgroundWords1 = new double[nWords];
    backgroundWords2 = new double[nWords];

    for (int w = 0; w < nWords; w ++){
       	  backgroundWords1[w] = 0;
       	  backgroundWords2[w] = 0;
       }

       for (std::vector<vote*>::iterator vi = trainVotes.begin(); vi != trainVotes.end(); vi++)
       {
         for (std::vector<int>::iterator it = (*vi)->words.begin(); it != (*vi)->words.end(); it++)
         {
           totalWords++;
           backgroundWords1[*it]++;
           backgroundWords2[*it]++;
         }
       }
       for (int w = 0; w < nWords; w++){
       	  backgroundWords1[w] /= totalWords;
       	  backgroundWords2[w] /= totalWords;
       }

       if (lambda1 == 0&&lambda2==0)
         {
           for (int u = 0; u < nUsers; u++)
           {
             if (nTrainingPerUser.find(u) == nTrainingPerUser.end()) continue;
             for (int k = 0; k < K; k++)
               gamma_user[u][k] = rand() * 1.0 / RAND_MAX;
           }
           for (int b = 0; b < nBeers; b++)
           {
             if (nTrainingPerBeer.find(b) == nTrainingPerBeer.end()) continue;
             for (int k = 0; k < K; k++)
               gamma_beer[b][k] = rand() * 1.0 / RAND_MAX;
           }
         }
         else
         {
           for (int w = 0; w < nWords; w++)
             for (int k = 0; k < K; k++){
             	 topicWords1[w][k] = 0;
             	 topicWords2[w][k] = 0;
             }

         }

    normalizeWordWeights();
    if (lambda1 > 0 || lambda2>0)
      updateTopics(true);

    *kappa1 = 1.0;
    *kappa2 = 1.0;
  }

  ~topicCorpus()
  {
	  delete[] votesPerBeer;
	     delete[] votesPerUser;
	     delete[] trainVotesPerBeer;
	     delete[] trainVotesPerUser;

	     for (int w = 0; w < nWords; w ++){
	     	 delete[] wordTopicCounts1[w];
	     	 delete[] wordTopicCounts2[w];
	     }

	     delete[] wordTopicCounts1;
	     delete[] wordTopicCounts2;

	     for (int b = 0; b < nBeers; b ++)
	       delete[] beerTopicCounts[b];
	     delete[] beerTopicCounts;
	     delete[] beerWords;
	     delete[] topicCounts1;
	     delete[] backgroundWords1;

	     for (int u = 0; u < nUsers; u ++)
	       delete[] userTopicCounts[u];
	     delete[] userTopicCounts;
	     delete[] userWords;
	     delete[] topicCounts2;
	     delete[] backgroundWords2;


	     for (std::vector<vote*>::iterator vi = trainVotes.begin(); vi != trainVotes.end(); vi++)
	     {
	       delete[] wordTopics1[*vi];
	       delete[] wordTopics2[*vi];
	     }

	     clearG(&alpha, &kappa1,&kappa2, &beta_user, &beta_beer, &gamma_user, &gamma_beer, &topicWords1,&topicWords2);
	     delete[] W;
  }

  double prediction(vote* vi);

  void dl(double* grad);
  void train(int emIterations, int gradIterations);
  double lsq(void);
  void validTestError(double& train, double& valid, double& test, double& testSte);
  void normalizeWordWeights(void);
  void save(char* modelPath, char* predictionPath);

  corpus* corp;
  
  // Votes from the training, validation, and test sets
  std::vector<vote*> trainVotes;
  std::vector<vote*> validVotes;
  std::set<vote*> testVotes;

  std::map<vote*, double> bestValidPredictions;

  std::vector<vote*>* votesPerBeer; // Vector of votes for each item
  std::vector<vote*>* votesPerUser; // Vector of votes for each user
  std::vector<vote*>* trainVotesPerBeer; // Same as above, but only votes from the training set
  std::vector<vote*>* trainVotesPerUser;

  int getG(double* g,
             double** alpha,
             double** kappa1,
  		     double** kappa2,
             double** beta_user,
             double** beta_beer,
             double*** gamma_user,
             double*** gamma_beer,
             double*** topicWords1,
  		     double*** topicWords2,
             bool init);
  void clearG(double** alpha,
               double** kappa1,
 			   double** kappa2,
               double** beta_user,
               double** beta_beer,
               double*** gamma_user,
               double*** gamma_beer,
               double*** topicWords1,
 			   double*** topicWords2);

  void wordZ1(double* res);
  void wordZ2(double* res);
  void topicZ(int beer, double& res);
  void topicZ1(int user, double& res);
  void updateTopics(bool sample);
  void topWords();

  // Model parameters
  double* alpha; // Offset parameter
  double* kappa1; // "peakiness" parameter
  double* kappa2; // "peakiness" parameter
  double* beta_user; // User offset parameters
  double* beta_beer; // Item offset parameters
  double** gamma_user; // User latent factors
  double** gamma_beer; // Item latent factors

  double* W; // Contiguous version of all parameters, i.e., a flat vector containing all parameters in order (useful for lbfgs)

  double** topicWords1; // Weights each word in each topic
  double** topicWords2; // Weights each word in each topic
  double* backgroundWords1; // "background" weight, so that each word has average weight zero across all topics
  double* backgroundWords2; // "background" weight, so that each word has average weight zero across all topics
  // Latent variables
  std::map<vote*, int*> wordTopics1;
  std::map<vote*, int*> wordTopics2;

  // Counters
  int** beerTopicCounts; // How many times does each topic occur for each product?
  int** userTopicCounts; // How many times does each topic occur for each user?
  int* beerWords; // Number of words in each "document"
  int* userWords; // Number of words in each user "document"

  long long* topicCounts1; // How many times does each topic occur?
  long long* topicCounts2; // How many times does each topic occur?
  int** wordTopicCounts1; // How many times does this topic occur for this word?
  int** wordTopicCounts2; // How many times does this topic occur for this word?

  long long totalWords; // How many words are there?

  int NW;
  int K;

  double latentReg;
  double lambda1;
  double lambda2;

  std::map<int,int> nTrainingPerUser; // Number of training items for each user
  std::map<int,int> nTrainingPerBeer; // and item

  int nUsers; // Number of users
  int nBeers; // Number of items
  int nWords; // Number of words
};
