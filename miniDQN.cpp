//		This program implements the minimax algorithm for playing tic-tac-toe, with
//		alpha-beta pruning. It also implements tabular Q-learning, and DQN.
//		Author: Ilia Smirnov

#include<iostream>
#include<fstream>
#include<sstream>
#include<algorithm>
#include<cstdlib>
#include<math.h>
#include<vector>
#include<tuple>
#include<time.h>
#include<string>
#include<random>
#include<queue>
#include<deque>
#include<iomanip>

using namespace std;
const int INFTY = 65536;

const int ROWS = 3;
const int COLMS = 3;
const int CELLS = ROWS * COLMS;
const int LINES = ROWS + COLMS + 2;
const int BOARDS = 19683;

enum strategy {MINIMAX, RANDOM, QTAB, HEURIBOT, DOUBLEQ, DQN};
strategy COMPSTRAT, COMPOPP;

const int EPOCHS = 1500000;			// Epochs of Tabular Q-training
const int EPOCHSTEP = 1000;			// How frequently training plot is updated

const float ITERS = 20000.0;
const float NUMBATTLES = 100000.0;		// Number of games in a battle

const int EPISODES = 350;
const int STEP = 1;
const bool QRAND = false;			// Randomize initial Q-values?
bool DEPTHVERBOSE = true;	  		// Include outputs regarding game depth?

const int DQN_EPOCHS = 500000;
double DQN_ALPHA = 1e-4;
const int DQN_ALPHA_UPDATE_FREQ = 400000;
const double DQN_ALPHA_SCALE = 0.1;
const double DQN_GAMMA = 0.9;
const double DQN_EPSILON = 0.1;
const double LEAK = 0.01;
const int BUFFER_CAPACITY = 300000;
const int TARGET_UPDATE_FREQUENCY = 25000;
const int MINIBATCH_SIZE = 2;
const int LAYERS = 4;
int NODES_IN_LAYER[LAYERS] = {CELLS, 32, 32, CELLS+1};
const int DQN_K = 1;

const int INPUT_LAYER = 0;
const int OUTPUT_LAYER = LAYERS-1;

enum Opti_type {SGD, SGDM, ADAM, RMS_PROP};
const Opti_type OPTI_TYPE = ADAM;
const double SGD_MOMENTUM = 0.9;
const double RMS_PROP_MOMENTUM = 0.95;
const double ADAM_MOMENT1 = 0.9;
const double ADAM_MOMENT2 = 0.999;
const double ONE_MINUS_RMS_PROP_MOMENTUM = 1 - RMS_PROP_MOMENTUM;
const double ONE_MINUS_ADAM_MOMENT1 = 1 - ADAM_MOMENT1;
const double ONE_MINUS_ADAM_MOMENT2 = 1 - ADAM_MOMENT2;
const double DQN_INIT_STDEV = 0.001;

class Board;
class Q_values;
class Node;
class Game_tree;
struct Orbit;
struct Tree_crawler;

class Q_Trainer;

struct wlt;
struct float_wlt;
class Battler;
class Exporter;

struct FF_input;
struct FF_node;
struct FF_layer;
class Neural_net;
class DQN_trainer;
double ReLU(double input);
double ReLUprime(double input);
double clip(double num, double low, double high);

//   =======================================================================
int line_cell_to_board_cell (int line, int line_cell);
int board_cell_to_line_cell (int cell, int line);

//   =======================================================================
// Passing between boards and hashes
int turn_to_ternary (int turn);
int lpos_to_hash(int lpos[]);

string strat2name (strategy strat);

enum optibot_strategy { WIN,
		        BLOCK_WIN,
			FORK,
			FORCE_BLOCK,
			BLOCK_FORK,
			CENTER_MOVE,
			OPP_CORNER,
			CORNER_MOVE,
			SIDE_MOVE,
			NO_MOVE };


// *********************************************************************************
// *********************************************************************************
//	  ******    ***          ****       *****     *****   *******     *****
//	***     **  ***         **  **     ***  **   ***  **  **         ***  **
//	**          ***        **    **      ***      ***     *****       ***
//	***     **  ***       **********  **  ***  **  ***    **       **  ***
//	  ******    *******  **        **  *****    *****     *******   *****
// *********************************************************************************
// *********************************************************************************

class Board {
private:
	int depth;
	int turn;				   // 1 = X to move;  -1 = O to move.
	int reward = 0;
	int winner = 0;
	bool terminal;
	int count(int pl);
	void check_win(int pl);
	int hash;
public:
	struct Line {
		int entry[3];
		int count_marks (int mark);
		bool spaces_free();
		bool check_one_of_three (int pl);
		bool check_two_of_three (int pl);
		int find_two_of_three (int pl);
		bool check_winning_line (int pl);

	};
	int lpos[CELLS];
	Line line[LINES];
	int get_depth();
	void set_depth(int d);
	int get_reward();
	int get_turn();
	int get_winner();
	bool cell_free(int i);
	bool is_terminal();
	void find_hash();
	int get_hash();
	void print();
	Board (int lpos[]);
	Board (int h = 0);
};

class Q_values {
public:
	double X[CELLS];
	double O[CELLS];
	void reset (bool r);
	Q_values (bool r = QRAND);
};

Q_values operator+ (Q_values, Q_values);

class Node {
private:
	// Minimax data
	int alpha = -INFTY;
	int beta = INFTY;
public:
	Board board;
	int hash = -1;
	bool children_were_generated = false;
	Node* children[CELLS];
	int children_cells[CELLS] = {0};
	int find_child_by_cell (int cell);
	void add_child (Node* child);
	void add_child (Board child_board);
	void add_child (int child_board_index);
	int last_child = -1;
	void compute_utility();
	int utility;
	int minimax_move;
	Q_values Q;
	double best_Q;
	void refresh_Q_data();
	void reset_Q_values(bool r);
	int TQ_move;
	Q_values Q1, Q2, Q_sum;
	int Q1_arg_optimal, Q2_arg_optimal;
	void refresh_double_Q_data();
	int find_Q_arg_optimal (Q_values* Q_pointer, int turn);
	void reset_double_Q_values(bool r);
	int double_TQ_move;
	int num_times_visited_by_Q_trainer = 0;
	int opti_move;
	optibot_strategy opti_strat;
	vector<int> opti_moves;
	int find_opti_move();
	void compute_opti_move();
	int DQ_move;
	double DQ_values[CELLS];
	int next_move (strategy strat);
	Node* next_move_node (strategy strat);
	Node();
	Node(Board init_board);
	Node(int init_board_index);
};

struct Orbit {
	Game_tree* tree;
	vector<int> orbit_hashes;
	int rep_hash;
	int size;
	void print();
	Orbit (Node* initial_node, Game_tree* tr);
	Orbit (int initial_node_hash, Game_tree* tr);
	Orbit (Board initial_board, Game_tree* tr);
};

struct Tree_crawler {
	Game_tree* tree;
	bool visited_node[BOARDS] = {false};
	void reset_visits();

	vector<Orbit> orbits;
	void find_orbit(Node* node);
	void find_orbits();				// Depth-first
	void find_orbits_breadthfirst();		// Breadth-first

	void compute_utility (Node* node);
	void compute_utilities ();
	void compute_opti_move (Node* node);
	void compute_opti_moves ();
	void reset_Q_value (Node* node, bool r = false);
	void reset_Q_values (bool r = false);
	void reset_double_Q_value(Node* node, bool r = false);
	void reset_double_Q_values(bool r);

	Tree_crawler(Game_tree* tr);
};

class Game_tree {
friend class Orbit;
friend class Tree_crawler;
friend class Exporter;
friend class Neural_net;
friend class DQN_trainer;
private:
	Node* table[BOARDS];
	bool htable[BOARDS] = {false};
	void gen_and_link(Node* parent);
	void hash(Node* node, int h);

public:
	Node root_node;
	void compute_utilities();
	void compute_opti_moves();
	void reset_Q_values(bool r);
	void reset_double_Q_values(bool r);

	Node* rotate_node_ccw (Node* initial_node);
	Node* rotate_node_ccw (Node* initial_node, int n);
	Node* reflect_node (Node* initial_node);
	Node* reflect_node (Node* initial_node, int n);

	Game_tree();
};

class Q_trainer {
private:
	// Q-learning parameters
	int cur_ep = 0;

	float initALPHA, initGAMMA, initEPSILON;

	int EXPLOREBDRY = EPOCHS / 3;
	int EXPLOREBDRYSTEP = EXPLOREBDRY / 9;
public:
	float ALPHA, GAMMA, EPSILON;
	Game_tree tree;
	void train(int eps = EPISODES);
	void train_double(int eps = EPISODES);
	void reset_Q();
	void reset_double_Q();
	void reset_hypers();
	wlt battle(strategy plstrat, strategy oppstrat, bool initrand = false);
	float_wlt battle(strategy plstrat, strategy oppstrat, int numit, bool initrand = false);
	string get_plot_data (int ep, int epstep, int numit, bool d = false, bool wrec = true, bool lrec = true, bool trec = true);
	string get_plot_data_avg_w_stdev (int ep, int epstep, int numit, bool d = false, bool wrec = true, bool lrec = true, bool trec = true);
	vector<string> get_plot_data_w_stdev (int ep, int epstep, int numit, bool d = false, bool wrec = true, bool lrec = true, bool trec = true);

	Q_trainer (float a=0.1, float g=0.9, float e=0.5);
	Q_trainer (float a, float g, float e, int epsbdry);
};

class Battler {
	wlt battle(strategy plstrat, strategy oppstrat);
	float_wlt battle(strategy plstrat, strategy oppstrat, int numit);

	string get_plot_data (int ep, int epstep, int numit, bool d = false, bool wrec = true, bool lrec = true, bool trec = true);
	string get_plot_data_avg_w_stdev (int ep, int epstep, int numit, bool d = false, bool wrec = true, bool lrec = true, bool trec = true);
	vector<string> get_plot_data_w_stdev (int ep, int epstep, int numit, bool d = false, bool wrec = true, bool lrec = true, bool trec = true);
};

struct wlt {
	int w = 0, l = 0, t = 0;
	int end_depths[CELLS+1]= {0};
	int win_depths[CELLS+1] = {0};

};

wlt operator+ (wlt, wlt);

struct float_wlt {
	float w = 0.0, l = 0.0, t = 0.0;
	float end_depths[CELLS+1] = {0.0};
	float win_depths[CELLS+1] = {0.0};
	float_wlt& operator= (const wlt& in);
};

float_wlt operator/ (float_wlt, int);

class Exporter {
private:
	Q_trainer Q = Q_trainer(1,0.9,1);
	Tree_crawler crawler = (&Q.tree);
	void print_board_to_html (Board board, ofstream& to, bool small = false);
	void print_board_to_string (Board board, ostringstream& tp, bool small = false);
	void print_minimax_board_to_html (Board board, ofstream& to, bool small = true);
	void print_Q_board_to_html (double lpos[], int turn, ofstream& to, bool small = true);
	void print_DQN_board_to_html (double DQ_values[], int turn, ofstream& to, double DQ_low, double DQ_range, bool small);
public:
	void output_html_strategy_table (char file_name[], bool breadth);
};

void fillQStrat(Node* node, vector<int>* strat_pointer, ofstream& to3);
void fill_strat(Game_tree& tree, int* strat_table, strategy STRAT);

struct FF_input {
	double value;
	double weight;
};

struct FF_node {
	int num_inputs = 0;
	FF_input* input = NULL;
	double value;
	double output;

	void compute();

	~FF_node();
};

struct FF_layer {
	int num_nodes = 0;
	FF_node* node = NULL;

	void enter(double inputs[]);
	void enter(int inputs[]);

	FF_layer();
	FF_layer(double inputs[], int num_nodes);
	FF_layer(int inputs[], int num_nodes);

	~FF_layer();
};

class Neural_net {
friend class DQN_trainer;
friend class DQN_trainer_with_two_nets;
private:
	static const int INPUT_LAYER_INDEX = 0;
	static const int OUTPUT_LAYER_INDEX = LAYERS - 1;
	FF_layer layer[LAYERS];
	FF_layer input_layer;
	FF_layer hidden_layer[LAYERS];
	FF_layer output_layer;
	int max_nodes;

	double ALPHA = DQN_ALPHA;

	// Optimizer data
	double s[4][128][128];
	double CUR_ADAM_MOMENT1 = ADAM_MOMENT1, CUR_ADAM_MOMENT2 = ADAM_MOMENT2;
	double m[4][128][128];
	double v[4][128][128];
	double m_est, v_est;
public:
	void input_game_node (Node* node);
	void forward_pass ();
	void backward_pass (int action[], double value[], double target[], int batch_size = MINIBATCH_SIZE);

	void fit_minimax(Game_tree* tree);
	void store_weights(ofstream& to);
	void load_weights(string& weight_str);
	void display();

	Neural_net();
};

struct Transition {
	Node* cur;
	Node* next;
	int a, r;
	Transition (Node* cur, Node* next, int a, int r);
};

struct Buffer {
	deque<Transition> transition;
	Game_tree* tree;
	void fill_both_sides(int capacity = BUFFER_CAPACITY);
	void fill_one_side(int capacity = BUFFER_CAPACITY, bool filling_O = false);
	Buffer(Game_tree* tree = NULL);
};

class DQN_trainer {
private:
	Neural_net trainer;
	vector<Neural_net> target;
	Game_tree* tree = NULL;
	int t = 0;
	double avg_Q[CELLS];
	Buffer replay_buffer;
	double GAMMA = DQN_GAMMA;
	double EPSILON = DQN_EPSILON;
	double max_DQ (int valid_cell[], Neural_net* action_net);
	double min_DQ (int valid_cell[], Neural_net* action_net);
	int max_DQ_action (int valid_cell[], Neural_net* net);
	int min_DQ_action (int valid_cell[], Neural_net* net);
public:
	double high_Q, low_Q;
	void fill_buffer (int capacity = BUFFER_CAPACITY);
	void train (int epochs = DQN_EPOCHS);
	void store_DQ_actions (Game_tree* tree);
	void find_Q_bounds();
	void store_weights (ofstream& to);
	void load_weights (ifstream& from);
	DQN_trainer (Game_tree* tree);
};

class DQN_trainer_with_two_nets {
private:
	Neural_net trainer_x, target_x, trainer_o, target_o;
	Game_tree* tree = NULL;
	int t = 0;
	double avg_Q[CELLS];
	Buffer replay_x, replay_o;
	double GAMMA = DQN_GAMMA;
	double EPSILON = DQN_EPSILON;
	int min_DQ_action(int valid_move[], Neural_net* net);
	int max_DQ_action(int valid_move[], Neural_net* net);
	double min_DQ (int valid_move[], Neural_net* net);
	double max_DQ (int valid_move[], Neural_net* net);
public:
	double high_Q, low_Q;
	void fill_buffers (int capacity = BUFFER_CAPACITY);
	void train (int epochs = DQN_EPOCHS);
	void store_DQ_actions (Game_tree* tree);
	void find_Q_bounds ();
	void store_weights (ofstream& to);
	void load_weights (ifstream& from);
	DQN_trainer_with_two_nets (Game_tree* tree);
};

typedef tuple<int,int,int> minimax_point;


// *********************************************************************************
// *********************************************************************************
// ******     ***       ***      ****      *********   ***     **     **************
// ******     ** **   ** **     **  **         *       ** **   **     **************
// ******     **  ** **  **    **    **        *       **  **  **     **************
// ******     **   ***   **   **********       *       **   ** **     **************
// ******     **         **  **        **  *********   **     ***     **************
// *********************************************************************************
// *********************************************************************************


int main(int argc, char* argv[]) {
	cout << fixed << setprecision(3);
	srand(time(NULL));

//====== DQN Training =====================================================
	ofstream to (argv[1]);
	ofstream weight_writer;

	Q_trainer Q, best;
	wlt xledger, oledger;
	DQN_trainer_with_two_nets tr(&Q.tree);
	float bestwinr = -INFTY, curwinr;
        std::string xrec = "", orec = "";

	const short sampling_rate = 100;
	for (int i = 0; i < DQN_EPOCHS; i += sampling_rate) {
		cout << i << endl;
		tr.train(sampling_rate);
		tr.store_DQ_actions(&Q.tree);
		xledger = Q.battle(DQN, RANDOM);
		std::cout << "W: " << xledger.w << "  L: " << xledger.l << "  T: " << xledger.t << std::endl;
                xrec += std::to_string(i) + " " + std::to_string(xledger.w) + ",";
		oledger = Q.battle(RANDOM, DQN);
		std::cout << "W: " << oledger.w << "  L: " << oledger.l << "  T: " << oledger.t << std::endl;
                orec += std::to_string(i) + " " + std::to_string(oledger.l) + ",";
                curwinr = (xledger.w + oledger.l) - (xledger.l + oledger.w);
                if (curwinr > bestwinr) {
			bestwinr = curwinr;
                	tr.store_DQ_actions(&best.tree);
			weight_writer = ofstream(argv[3]);
			tr.store_weights(weight_writer);
			weight_writer.close();
		}
	}
	to << xrec << ";" << orec << ";!";

	std::cout << "DQN v. minimax" << std::endl;
        best.battle(DQN, MINIMAX, 1000);
	std::cout << "minimax v. DQN" << std::endl;
        best.battle(MINIMAX, DQN, 1000);
	std::cout << "DQN v. random" << std::endl;
        best.battle(DQN, RANDOM, 1000);
	std::cout << "random v. DQN" << std::endl;
        best.battle(RANDOM, DQN, 1000);

	to << "Alpha: " << DQN_ALPHA << endl;
	to << "Gamma: " << DQN_GAMMA << endl;
	to << "Epsilon: " << DQN_EPSILON << endl;
	to << "Minibatch: " << MINIBATCH_SIZE << endl;
	to << "Buffer capacity: " << BUFFER_CAPACITY << endl;
	to << "Leak: " << LEAK << endl;
	to << "Target update freq.: " << TARGET_UPDATE_FREQUENCY << endl;
	to << "No. layers: " << LAYERS << endl;
	for (int i = 0; i < LAYERS; i++) {
		to << NODES_IN_LAYER[i] << " " ;
	}
        to << endl;
        to << "Optimizer: " << OPTI_TYPE;
	to.close();

// ==== Strategy export =============================================================
	ofstream wr(argv[2]);
	int strat_table[BOARDS];
        fill_strat(best.tree, strat_table, DQN);
        for (int i = 0; i < BOARDS; i++)
            wr << strat_table[i] << " ";
        wr << ".";
        wr.close();

/*
//======= Battling every pair of agents ===================================
	Q_trainer Q(1, 0.999, 1);
	DQN_trainer_with_two_nets tr(&Q.tree);
	std::ifstream from("weights1280.txt");
	tr.load_weights(from);
	Tree_crawler crawler = Tree_crawler(&Q.tree);

						// ...Load (or train)...
	Q.tree.compute_utilities();		// Minimax strategy
	tr.store_DQ_actions(&Q.tree);		// DQN strategy
	crawler.compute_opti_moves();		// Heuribot strategy
	Q.train(5000000);			// Single Q-learning (tabular) strategy
        Q.train_double(5000000);		// Double Q-learning (tabular) strategy

	const int num_agents = 6;
	strategy pl_strat, opp_strat;
	for (int pl = 0; pl < num_agents; pl++) {
		for (int opp = 0; opp < num_agents; opp++) {
			pl_strat = static_cast<strategy>(pl);
			opp_strat = static_cast<strategy>(opp);
			std::cout << strat2name(pl_strat) << "  v.  " << strat2name(opp_strat) << "\t";
			Q.battle(pl_strat, opp_strat, 1000);
                        std::cout << std::endl;
		}
	}
	for (int pl = 0; pl < num_agents; pl++) {
		for (int opp = 0; opp < num_agents; opp++) {
			pl_strat = static_cast<strategy>(pl);
			opp_strat = static_cast<strategy>(opp);
			std::cout << strat2name(pl_strat) << "  v.  " << strat2name(opp_strat) << "\t";
			Q.battle(pl_strat, opp_strat, 1000, true);
                        std::cout << std::endl;
		}
	}
*/
/*
// ===== Generate strategy tables ===================================================
	Exporter exp;
	exp.output_html_strategy_table(argv[1], false);
*/

/*
// =========== Recording performance of Tabular Q ==========================
	Q_trainer Q(1, 0.9, 1);
	COMPSTRAT = DOUBLEQ;
	COMPOPP = RANDOM;

	Tree_crawler cr(&Q.tree);
	if (COMPOPP == MINIMAX) cr.compute_utilities();
	if (COMPOPP == HEURIBOT) cr.compute_opti_moves();
	if (COMPOPP == QTAB) { Q.reset_Q(); Q.reset_hypers(); Q.train(5000000); }
	if (COMPOPP == DOUBLEQ) { Q.reset_double_Q(); Q.reset_hypers(); Q.train_double(5000000); }

	string record = Q.get_plot_data_avg_w_stdev (EPOCHS, EPOCHSTEP, 50,
                                                     // Should DOUBLEQ or QTAB be trained?
                                                     COMPSTRAT == DOUBLEQ,
                                                     // Record wins?
                                                     COMPOPP == RANDOM,
                                                     // Record losses?
                                                      false,
                                                     // Record ties?
                                                      COMPOPP != RANDOM);

	cout << Q.ALPHA << endl;
	cout << Q.GAMMA << endl;
	cout << Q.EPSILON << endl;

	ofstream to (argv[1]);
	to << record;
	to << "!";
	to.close();
*/
	return 0;
}

// *********************************************************************************
// *********************************************************************************
// **************** Neural network classes *****************************************

void FF_node::compute() {
	value = 0.0;
	for (int w = 0; w < num_inputs; w++) {
		value += input[w].value * input[w].weight;
	}
	output = ReLU(value);
}

FF_node::~FF_node() {
	//delete []input;
	input = NULL;
}

FF_layer::FF_layer(){
}

FF_layer::FF_layer(int value[], int num_nodes) {
	node = new FF_node[num_nodes];
	for (int n = 0; n < num_nodes; n++) node[n].value = value[n];
}

FF_layer::FF_layer(double value[], int num_nodes) {
	node = new FF_node[num_nodes];
	for (int n = 0; n < num_nodes; n++) node[n].value = value[n];
}

void FF_layer::enter (int value[]) {
	for (int n = 0; n < num_nodes; n++) node[n].value = value[n];
}

void FF_layer::enter (double value[]) {
	for (int n = 0; n < num_nodes; n++) node[n].value = value[n];
}

FF_layer::~FF_layer() {
	//delete []node;
	node = NULL;
}

Neural_net::Neural_net() {
	default_random_engine gen;
	normal_distribution<double> dist(0, DQN_INIT_STDEV);

	for (int l = 0; l < LAYERS; l++) {
		layer[l].num_nodes = NODES_IN_LAYER[l];
		layer[l].node = new FF_node[layer[l].num_nodes];

		if (l != INPUT_LAYER) {
			for (int n = 0; n < layer[l].num_nodes; n++) {
				layer[l].node[n].value = dist(gen);
				layer[l].node[n].num_inputs = layer[l-1].num_nodes;
				layer[l].node[n].input = new FF_input[layer[l].node[n].num_inputs];

				for (int w = 0; w < layer[l].node[n].num_inputs; w++) {
					layer[l].node[n].input[w].value = dist(gen);
					layer[l].node[n].input[w].weight = dist(gen);
				}
			}
			// Set up the bias node
				layer[l].node[ layer[l].num_nodes-1 ].value = 0;
				layer[l].node[ layer[l].num_nodes-1 ].output = 1.0;
	}	}
	max_nodes = 0;
	for (int l = 0; l < LAYERS; l++) {
		if (NODES_IN_LAYER[l] > max_nodes) max_nodes = NODES_IN_LAYER[l];
	}

	for (int l = 0; l < 4; l++) {
		for (int n = 0; n < layer[l].num_nodes; n++) {
			for (int w = 0; w < layer[l].node[w].num_inputs; w++) {
				s[l][n][w] = 0.0;
				v[l][n][w] = 0.0;
                                m[l][n][w] = 0.0;
	} 	}	 }

}

void Neural_net::display() {
	for (int r = 0; r < max_nodes; r++) {
		for (int l = 0; l < LAYERS; l++) {
			if (r < layer[l].num_nodes) cout << layer[l].node[r].value << " (" << layer[l].node[r].output << "), ";
			else cout << "      N,      ";
		}
		cout << endl;
	}
}

void Neural_net::input_game_node (Node* node) {
	layer[INPUT_LAYER_INDEX].enter ( node->board.lpos );
}

void Neural_net::forward_pass(){
	// Prepare input layer
	for (int n = 0; n < layer[INPUT_LAYER_INDEX].num_nodes; n++) {
		layer[INPUT_LAYER_INDEX].node[n].output = layer[INPUT_LAYER_INDEX].node[n].value;
	}
	// Copy outputs of previous layer to inputs of previous layer and compute node values
	for (int l = 1; l < LAYERS; l++) {
		for (int n = 0; n < layer[l].num_nodes-1; n++) {
			for (int w = 0; w < layer[l].node[n].num_inputs; w++) {
				layer[l].node[n].input[w].value = layer[l-1].node[w].output;
			}
			layer[l].node[n].compute();
		}
	}
}

void Neural_net::backward_pass(int a[], double value[], double target[], int batch_size){
	double delta[LAYERS][max_nodes];
	double dw[LAYERS][max_nodes][max_nodes];
	double diff = 0.0;

	for (int l = 0; l < LAYERS; l++) {
		for (int n = 0; n < layer[l].num_nodes; n++) {
			for (int w = 0; w < layer[l].node[n].num_inputs; w++) {
				dw[l][n][w] = 0.0;
			}
		}
	}

	// Compute error gradients of the minibatch
	for (int batch = 0; batch < batch_size; batch++) {

		for (int l = 0; l < LAYERS; l++) {
			for (int n = 0; n < layer[l].num_nodes; n++) {
				delta[l][n] = 0.0;
		}	}

		delta[OUTPUT_LAYER][a[batch]] = -(target[batch] - value[batch]);
	        delta[OUTPUT_LAYER][a[batch]] = clip( delta[OUTPUT_LAYER][a[batch]], -1.0, 1.0);
		// Back-propagate
		for (int l = OUTPUT_LAYER - 1; l > 0; l--) {
			for (int n = 0; n < layer[l].num_nodes; n++) {
				for (int m = 0; m < layer[l+1].num_nodes; m++) {
					delta[l][n] += layer[l+1].node[m].input[n].weight  * delta[l+1][m];
				}
				delta[l][n] *= ReLUprime( layer[l].node[n].value );
			}
		}

		for (int l = 1; l < LAYERS; l++) {
			for (int n = 0; n < layer[l].num_nodes; n++) {
				for (int w = 0; w < layer[l].node[n].num_inputs; w++) {
					dw[l][n][w] += layer[l-1].node[w].output * delta[l][n]  * ((double) 1/MINIBATCH_SIZE);
				}
			}
		}
	}

	if (OPTI_TYPE == ADAM) {
		CUR_ADAM_MOMENT1 *= ADAM_MOMENT1;
		CUR_ADAM_MOMENT2 *= ADAM_MOMENT2;
	}

	for (int l = 1; l < LAYERS; l++) {
		for (int n = 0; n < layer[l].num_nodes; n++) {
			for (int w = 0; w < layer[l].node[n].num_inputs; w++) {
				//dw[l][n][w] = clip(dw[l][n][w], -1.0, 1.0);

				switch(OPTI_TYPE) {
					case SGD    :    layer[l].node[n].input[w].weight -= ALPHA * dw[l][n][w];
                                                         break;
                                        case SGDM   :    s[l][n][w] = SGD_MOMENTUM * s[l][n][w] + ALPHA * dw[l][n][w];
                                                         layer[l].node[n].input[w].weight -= s[l][n][w];
                                                         break;
                                        case RMS_PROP    :    s[l][n][w] = RMS_PROP_MOMENTUM * s[l][n][w] + ONE_MINUS_RMS_PROP_MOMENTUM*dw[l][n][w]*dw[l][n][w];
					                      layer[l].node[n].input[w].weight -= ALPHA * (1/(sqrt(s[l][n][w]+0.01))) * dw[l][n][w];
							      break;
					case ADAM    :	  m[l][n][w] = ADAM_MOMENT1 * m[l][n][w] + ONE_MINUS_ADAM_MOMENT1 * dw[l][n][w];
					                  v[l][n][w] = ADAM_MOMENT2 * v[l][n][w] + ONE_MINUS_ADAM_MOMENT2 * dw[l][n][w] * dw[l][n][w];
	   	                          		  m_est = m[l][n][w] / (1 - CUR_ADAM_MOMENT1);
					                  v_est = v[l][n][w] / (1 - CUR_ADAM_MOMENT2);
 				                          layer[l].node[n].input[w].weight -= ALPHA * m_est / (sqrt(v_est) + 0.000001);
                                                          break;
				}
			}
		}
	}
}

double ReLU(double input) {
	if (input < 0) return LEAK * input;
	else return input;
}

double ReLUprime(double input) {
	if (input < 0) return LEAK;
	else return 1;
}

double clip (double num, double low, double high) {
	num = max(low, num);
	num = min(high, num);
	return num;
}

void Neural_net::store_weights(ofstream& to) {
	for (int l = 1; l < LAYERS; l++) {
		for (int n = 0; n < NODES_IN_LAYER[l]; n++) {
			for (int m = 0; m < NODES_IN_LAYER[l-1]; m++) {
				to << std::to_string(layer[l].node[n].input[m].weight) << " ";
			}
		}
	}
	to << ";";
}

void Neural_net::load_weights(string& input) {
	string weight_string = "";
	int num_weights = 0;
	for (int l = 1; l < LAYERS; l++)
		num_weights += NODES_IN_LAYER[l-1] * NODES_IN_LAYER[l];
	vector<float> weights(num_weights);
	vector<float>::iterator wit = weights.begin();
	int i = 0;
	char ch;
	while ((ch = input[i]) != ';') {
		switch(ch) {
			case ' '    :    *wit = stof(weight_string);
					 wit++;
					 weight_string = "";
					 break;
			default     :    weight_string += ch;
					 break;
		}
		i++;
	}

	int weight_counter = 0;
	for (int l = 1; l < LAYERS; l++) {
		for (int n = 0; n < NODES_IN_LAYER[l]; n++) {
			for (int m = 0; m < NODES_IN_LAYER[l-1]; m++) {
				layer[l].node[n].input[m].weight = weights[weight_counter];
				weight_counter++;
			}
		}
	}
}

void Neural_net::fit_minimax(Game_tree* tree) {
	Tree_crawler crawler = Tree_crawler(tree);
	crawler.compute_utilities();
	Node* node;
	queue<Node*> node_queue;

	vector<minimax_point> data, epoch;
	auto rng = default_random_engine {};
	crawler.reset_visits();
	node = &(tree->root_node);
	node_queue.push(node);
	crawler.visited_node[0] = true;
	int cr = 0;

	while (node_queue.size() > 0) {
		node = node_queue.front();
		node_queue.pop();

		if (!node->board.is_terminal()) {
			for (int i = 0; i < CELLS; i++) {
				if (node->board.lpos[i] == 0) {
					data.push_back( make_tuple(node->hash, i, node->children[node->find_child_by_cell(i)]->utility) );
				}
			}

			for (int c = 0; c <= node->last_child; c++)
				if (!crawler.visited_node[node->children[c]->hash]) {
					node_queue.push(node->children[c]);
					crawler.visited_node[node->children[c]->hash] = true;
				}
		}
	}

	int batch_draw;
	minimax_point data_point;
	int hash, a, utility;
	int action_batch[MINIBATCH_SIZE];
	double DQ_value_batch[MINIBATCH_SIZE];
	double target_batch[MINIBATCH_SIZE];

	double loss, diff, best_loss = INFTY;

	for (int ep = 0; ep < 2000; ep++) {
		epoch = data;
		shuffle(epoch.begin(), epoch.end(), rng);

		while ( epoch.size() > MINIBATCH_SIZE ) {

			for (int b = 0; b < MINIBATCH_SIZE; b++) {
				batch_draw = rand() % epoch.size();
				data_point = epoch[batch_draw];
				epoch.erase(epoch.begin() + batch_draw);
				hash = get<0>(data_point);
				a = get<1>(data_point);
				utility = get<2>(data_point);

				node = tree->table[hash];

				layer[INPUT_LAYER].enter(node->board.lpos);
				forward_pass();
				//display();

				action_batch[b] = a;
				DQ_value_batch[b] = layer[OUTPUT_LAYER].node[a].value;
				target_batch[b] = utility;
			}
			backward_pass(action_batch, DQ_value_batch, target_batch);
		}
		crawler.reset_visits();
		node = &(tree->root_node);
		node_queue.push(node);
		crawler.visited_node[0] = true;
		loss = 0.0;
		while (node_queue.size() > 0) {
			node = node_queue.front();
			node_queue.pop();
			if (!node->board.is_terminal()) {
				layer[INPUT_LAYER].enter(node->board.lpos);
				forward_pass();
				for (int i = 0; i < CELLS; i++) {
					if (node->board.lpos[i] == 0) {
						diff = node->children[node->find_child_by_cell(i)]->utility - layer[OUTPUT_LAYER].node[i].value;
						loss += diff*diff;
					}
				}
				for (int c = 0; c <= node->last_child; c++) {
					if (!crawler.visited_node[node->children[c]->hash]) {
						node_queue.push(node->children[c]);
						crawler.visited_node[node->children[c]->hash] = true;
					}
				}
			}
		}
		if (loss < best_loss) best_loss = loss;

		cout << "Epoch: " << ep << "; Loss: " << loss << endl;

		if (ep == 350) DQN_ALPHA = 0.001;

	}

	cout << "Best loss: " << best_loss << endl;

}

// ==============================================================================
// ========================== DQN Trainer =======================================

Transition::Transition (Node* cur, Node* next, int a, int r) {
	this->cur = cur;
	this->next = next;
	this->a = a;
	this->r = r;
}

Buffer::Buffer (Game_tree* tree) {
	this->tree = tree;
}

void Buffer::fill_both_sides (int capacity) {
	int n = 0, a, b, r;
	Node *cur = &(tree->root_node), *mid, *next;
	Transition trans = Transition(cur, cur, 0, 0);

	while (n < capacity) {
		cur = &(tree->root_node);
		a = rand() % (cur->last_child + 1);
		mid = cur->children[a];
		while (!cur->board.is_terminal()) {
			if (!mid->board.is_terminal()) {
				b = rand() % (mid->last_child + 1);
				next = mid->children[b];
			}
			else next = mid;

			trans = Transition(cur, next, a, next->board.get_reward());
			transition.push_back(trans);
			cur = mid;  mid = next;  a = b;
			n++;
		}
	}
}

void Buffer::fill_one_side (int capacity, bool filling_O) {
	int n = 0, a, b, r;
	Node *cur = &(tree->root_node), *mid, *next;
	Transition trans(cur, cur, 0, 0);

	while (n < capacity) {
		cur = &(tree->root_node);
		if (filling_O) cur = cur->children[rand() % CELLS];
		while (!cur->board.is_terminal()) {
			a = rand() % (cur->last_child + 1);
			mid = cur->children[a];

			if (!mid->board.is_terminal()) {
				b = rand() % (mid->last_child + 1);
				next = mid->children[b];
			}
			else next = mid;

			trans = Transition(cur, next, a, next->board.get_reward());
			transition.push_back(trans);
			cur = next;
			n++;
		}
	}
}

void DQN_trainer::store_DQ_actions (Game_tree* tree) {
	Tree_crawler crawler = Tree_crawler(tree);
	crawler.reset_visits();

	Node* node = &(tree->root_node);
	queue<Node*> node_queue;
	node_queue.push(node);
	crawler.visited_node[0] = true;

	while (node_queue.size() > 0) {
		node = node_queue.front();
		node_queue.pop();
		if (!node->board.is_terminal()) {
			trainer.layer[INPUT_LAYER].enter(node->board.lpos);
			trainer.forward_pass();
			node->DQ_move = node->find_child_by_cell(
	                                        node->board.get_turn() == 1 ?
  						max_DQ_action(node->board.lpos, &trainer) :
						min_DQ_action(node->board.lpos, &trainer)
                                        );
			for (int i = 0; i <= node->last_child; i++) {
				if (!crawler.visited_node[node->children[i]->hash]) {
					node_queue.push(node->children[i]);
					crawler.visited_node[node->children[i]->hash] = true;
				}
			}
		}
	}
}

void DQN_trainer::fill_buffer (int capacity) {
	replay_buffer.fill_both_sides(capacity);
}

DQN_trainer::DQN_trainer (Game_tree* tree) {
	this->tree = tree;
	replay_buffer = Buffer(tree);
	replay_buffer.fill_both_sides();

	Neural_net init;
	for (int i = 0; i < DQN_K; i++) {
		init = Neural_net();	// Initialize with random weights
		target.push_back(init);
	}
}

void DQN_trainer::train (int epochs) {
	double random;
	random_device rdev;
	mt19937 gen(rdev());
	uniform_real_distribution<> dis(0, 1.0);

	Node *cur = &(tree->root_node), *mid, *next;
	int a, b;
	Transition trans = Transition(cur, cur, 0, 0);
	double DQ_target;
	int batch_draw;
	int action_batch[MINIBATCH_SIZE];
	double DQ_value_batch[MINIBATCH_SIZE];
	double target_batch[MINIBATCH_SIZE];

	for (int ep = 0; ep < epochs; ep++) {
		cur = &(tree->root_node);
                if (dis(gen) < EPSILON) a = rand() % (cur->last_child + 1);
		else a = cur->find_child_by_cell(max_DQ_action(cur->board.lpos, &trainer));
		mid = cur->children[a];

		while(!cur->board.is_terminal()) {
			if (!mid->board.is_terminal()) {
				if (dis(gen) < EPSILON) b = rand() % (mid->last_child + 1);
				else mid->board.get_turn() == 1 ?
					b = mid->find_child_by_cell(max_DQ_action(mid->board.lpos, &trainer)) :
					b = mid->find_child_by_cell(min_DQ_action(mid->board.lpos, &trainer));
				next = mid->children[b];
			}
			else next = mid;

			trans = Transition(cur, next, a, next->board.get_reward());
			replay_buffer.transition.pop_front();
			replay_buffer.transition.push_back(trans);
			cur = mid;  mid = next;  a = b;

			for (int batch = 0; batch < MINIBATCH_SIZE; batch++) {
				batch_draw = rand() % BUFFER_CAPACITY;
				trans = replay_buffer.transition[batch_draw];

				if (trans.next->board.is_terminal()) DQ_target = trans.r;
				else {
					trainer.layer[INPUT_LAYER].enter(trans.next->board.lpos);
					trainer.forward_pass();
					DQ_target = GAMMA *
						    (trans.cur->board.get_turn() == 1 ?
						     max_DQ(trans.next->board.lpos, &trainer) :
						     min_DQ(trans.next->board.lpos, &trainer));
				}
				trainer.layer[INPUT_LAYER].enter(trans.cur->board.lpos);
				trainer.forward_pass();
				action_batch[batch] = trans.cur->children_cells[trans.a];
				DQ_value_batch[batch] = trainer.layer[OUTPUT_LAYER].node[action_batch[batch]].value;
				target_batch[batch] = DQ_target;
			}
			trainer.backward_pass(action_batch, DQ_value_batch, target_batch);
		}
		if (ep % TARGET_UPDATE_FREQUENCY == 0) {
			target.erase(target.begin());			// Using averaged DQN
			target.push_back(trainer);
		}
		t++;
                if (t % (DQN_ALPHA_UPDATE_FREQ) == 0) trainer.ALPHA *= 0.1;
	}
}

double DQN_trainer::max_DQ (int lpos[], Neural_net* action_net) {
	for (int cell; cell < CELLS; cell++)
		avg_Q[cell] = 0.0;
	for (int i = 0; i < DQN_K; i++) {
		target[i].layer[INPUT_LAYER].enter(lpos);
		target[i].forward_pass();
		for (int cell; cell < CELLS; cell++)
			avg_Q[cell] += target[i].layer[OUTPUT_LAYER].node[cell].value;
	}
	for (int cell = 0; cell < CELLS; cell++)
		avg_Q[cell] *= ((double) 1/DQN_K );

	return avg_Q [ max_DQ_action(lpos, action_net) ];
}

double DQN_trainer::min_DQ (int lpos[], Neural_net* action_net) {
	for (int cell; cell < CELLS; cell++)
		avg_Q[cell] = 0.0;
	for (int i = 0; i < DQN_K; i++) {
		target[i].layer[INPUT_LAYER].enter(lpos);
		target[i].forward_pass();
		for (int cell; cell < CELLS; cell++)
			avg_Q[cell] += target[i].layer[OUTPUT_LAYER].node[cell].value;
	}
	for (int cell = 0; cell < CELLS; cell++)
		avg_Q[cell] *= ((double) 1/DQN_K);
	return avg_Q[min_DQ_action(lpos, action_net)];
}

int DQN_trainer::max_DQ_action (int lpos[], Neural_net* net) {
	double max = -INFTY;
	int max_action = 0;

	net->layer[INPUT_LAYER].enter(lpos);
	net->forward_pass();

	for (int i = 0; i < CELLS; i++) {
		if (lpos[i] == 0) {
			if (net->layer[OUTPUT_LAYER].node[i].value > max) {
				max = net->layer[OUTPUT_LAYER].node[i].value;
				max_action = i;
			}
		}
	}
	return max_action;
}

int DQN_trainer::min_DQ_action (int lpos[], Neural_net* net) {
	double min = INFTY;
	int min_action = 0;

	net->layer[INPUT_LAYER].enter(lpos);
	net->forward_pass();

	for (int i = 0; i < CELLS; i++) {
		if (lpos[i] == 0) {
			if (net->layer[OUTPUT_LAYER].node[i].value < min) {
				min = net->layer[OUTPUT_LAYER].node[i].value;
				min_action = i;
			}
		}
	}
	return min_action;
}

void DQN_trainer::load_weights(ifstream& from) {
	string weights;
	getline(from, weights);
	trainer.load_weights(weights);
}

void DQN_trainer::store_weights(ofstream& to) {
	trainer.store_weights(to);
}


DQN_trainer_with_two_nets::DQN_trainer_with_two_nets (Game_tree* tree) {
	this->tree = tree;
	replay_x = Buffer(tree);
	replay_o = Buffer(tree);
	fill_buffers();
}

int DQN_trainer_with_two_nets::max_DQ_action(int lpos[], Neural_net* net) {
	double max = -INFTY;
	int max_a;

	net->layer[INPUT_LAYER].enter(lpos);
	net->forward_pass();

	for (int i = 0; i < CELLS; i++) {
		if (!lpos[i]) {
			if (net->layer[OUTPUT_LAYER].node[i].value > max) {
				max = net->layer[OUTPUT_LAYER].node[i].value;
				max_a = i;
			}
		}
	}
	return max_a;
}

int DQN_trainer_with_two_nets::min_DQ_action(int lpos[], Neural_net* net) {
	double min = INFTY;
	int min_a;

	net->layer[INPUT_LAYER].enter(lpos);
	net->forward_pass();

	for (int i = 0; i < CELLS; i++) {
		if (!lpos[i]) {
			if (net->layer[OUTPUT_LAYER].node[i].value < min) {
				min = net->layer[OUTPUT_LAYER].node[i].value;
				min_a = i;
			}
		}
	}
	return min_a;
}

double DQN_trainer_with_two_nets::max_DQ (int lpos[], Neural_net* net) {
	net->layer[INPUT_LAYER].enter(lpos);
	net->forward_pass();
	return net->layer[OUTPUT_LAYER].node[max_DQ_action(lpos, net)].value;
}

double DQN_trainer_with_two_nets::min_DQ (int lpos[], Neural_net* net) {
	net->layer[INPUT_LAYER].enter(lpos);
	net->forward_pass();
	return net->layer[OUTPUT_LAYER].node[min_DQ_action(lpos, net)].value;
}

void DQN_trainer_with_two_nets::train(int epochs) {
	double random;
	random_device rdev;
	mt19937 gen(rdev());
	uniform_real_distribution<> dis(0, 1.0);

	Node *cur = &(tree->root_node), *mid, *next;
	int a, b;
	Transition trans(cur, cur, 0, 0);
	Transition trans_x = trans, trans_o = trans;
	double DQ_target_x, DQ_target_o;
	int batch_draw;
	int action_batch_x[MINIBATCH_SIZE], action_batch_o[MINIBATCH_SIZE];
	double DQ_value_batch_x[MINIBATCH_SIZE], DQ_value_batch_o[MINIBATCH_SIZE];
	double target_batch_x[MINIBATCH_SIZE], target_batch_o[MINIBATCH_SIZE];

	for (int ep = 0; ep < epochs; ep++) {
		cur = &(tree->root_node);
                if (dis(gen) < EPSILON) a = rand() % (cur->last_child + 1);
		else a = cur->find_child_by_cell(max_DQ_action(cur->board.lpos, &trainer_x));
		mid = cur->children[a];

		while(!cur->board.is_terminal()) {
			if (!mid->board.is_terminal()) {
				if (dis(gen) < EPSILON) b = rand() % (mid->last_child + 1);
				else mid->board.get_turn() == 1 ?
					b = mid->find_child_by_cell(max_DQ_action(mid->board.lpos, &trainer_x)) :
					b = mid->find_child_by_cell(min_DQ_action(mid->board.lpos, &trainer_o));
				next = mid->children[b];
			}
			else next = mid;

			trans = Transition(cur, next, a, next->board.get_reward());
			if (cur->board.get_turn() == 1) {
				replay_x.transition.pop_front();
				replay_x.transition.push_back(trans);
			}
			else {
				replay_o.transition.pop_front();
				replay_o.transition.push_back(trans);
			}
			cur = mid;  mid = next;  a = b;

			for (int batch = 0; batch < MINIBATCH_SIZE; batch++) {
				batch_draw = rand() % BUFFER_CAPACITY;
				trans_x = replay_x.transition[batch_draw];
				trans_o = replay_o.transition[batch_draw];

				if (trans_x.next->board.is_terminal()) DQ_target_x = trans_x.r;
				else {
					b = max_DQ_action(trans_x.next->board.lpos, &target_x);    // Double DQN
					trainer_x.layer[INPUT_LAYER].enter(trans_x.next->board.lpos);
					trainer_x.forward_pass();
					DQ_target_x = GAMMA * trainer_x.layer[OUTPUT_LAYER].node[b].value;
				}
				trainer_x.layer[INPUT_LAYER].enter(trans_x.cur->board.lpos);
				trainer_x.forward_pass();
				action_batch_x[batch] = trans_x.cur->children_cells[trans_x.a];
				DQ_value_batch_x[batch] = trainer_x.layer[OUTPUT_LAYER].node[action_batch_x[batch]].value;
				target_batch_x[batch] = DQ_target_x;

				if (trans_o.next->board.is_terminal()) DQ_target_o = trans_o.r;
				else {
					b = min_DQ_action(trans_o.next->board.lpos, &target_o);
					trainer_o.layer[INPUT_LAYER].enter(trans_o.next->board.lpos);
					trainer_o.forward_pass();
					DQ_target_o = GAMMA * trainer_o.layer[OUTPUT_LAYER].node[b].value;
				}
				trainer_o.layer[INPUT_LAYER].enter(trans_o.cur->board.lpos);
				trainer_o.forward_pass();
				action_batch_o[batch] = trans_o.cur->children_cells[trans_o.a];
				DQ_value_batch_o[batch] = trainer_o.layer[OUTPUT_LAYER].node[action_batch_o[batch]].value;
				target_batch_o[batch] = DQ_target_o;
			}
			trainer_x.backward_pass(action_batch_x, DQ_value_batch_x, target_batch_x);
			trainer_o.backward_pass(action_batch_o, DQ_value_batch_o, target_batch_o);
		}
		if (ep % TARGET_UPDATE_FREQUENCY == 0) {
			target_x = trainer_x; target_o = trainer_o;
		}
		t++;
                if (t % (DQN_ALPHA_UPDATE_FREQ) == 0) {
			trainer_x.ALPHA *= DQN_ALPHA_SCALE; trainer_o.ALPHA *= DQN_ALPHA_SCALE;
		}
	}
}

void DQN_trainer_with_two_nets::store_DQ_actions (Game_tree* tree) {
	Tree_crawler crawler = Tree_crawler(tree);
	crawler.reset_visits();

	Node* node = &(tree->root_node);
	queue<Node*> node_queue;
	node_queue.push(node);
	crawler.visited_node[0] = true;
	int cell;

	while (node_queue.size() > 0) {
		node = node_queue.front();
		node_queue.pop();
		if (!node->board.is_terminal()) {
			if (node->board.get_turn() == 1) {
				trainer_x.layer[INPUT_LAYER].enter(node->board.lpos);
				trainer_x.forward_pass();
				node->DQ_move = node->find_child_by_cell( max_DQ_action(node->board.lpos, &trainer_x) );
			}
			else {
				trainer_o.layer[INPUT_LAYER].enter(node->board.lpos);
				trainer_o.forward_pass();
				node->DQ_move = node->find_child_by_cell( min_DQ_action(node->board.lpos, &trainer_o) );
			}
			for (int i = 0; i <= node->last_child; i++) {
				cell = node->children_cells[i];
				node->DQ_values[cell] = node->board.get_turn() == 1 ?
						      trainer_x.layer[OUTPUT_LAYER].node[cell].value :
						      trainer_o.layer[OUTPUT_LAYER].node[cell].value ;
				if (!crawler.visited_node[node->children[i]->hash]) {
					node_queue.push(node->children[i]);
					crawler.visited_node[node->children[i]->hash] = true;
				}
			}
		}
	}
}

void DQN_trainer_with_two_nets::fill_buffers (int capacity) {
	replay_x.fill_one_side(capacity, false);
	replay_o.fill_one_side(capacity, true);
}

void DQN_trainer_with_two_nets::store_weights (ofstream& to) {
	trainer_x.store_weights(to);
	trainer_o.store_weights(to);
}

void DQN_trainer_with_two_nets::load_weights (ifstream& from) {
	string weights, xweights, oweights;
	getline(from, weights);
	size_t semicolon = weights.find(';');
	xweights = weights.substr(0, semicolon + 1);
	oweights = weights.substr(semicolon + 1);
	trainer_x.load_weights(xweights);
	trainer_o.load_weights(oweights);
}

void DQN_trainer_with_two_nets::find_Q_bounds () {
	Tree_crawler crawler = Tree_crawler(tree);
	crawler.reset_visits();

	Node* node = &(tree->root_node);
	queue<Node*> node_queue;
	node_queue.push(node);
	crawler.visited_node[0] = true;

	double cur_Q;
	high_Q = -INFTY; low_Q = INFTY;

	while (node_queue.size() > 0) {
		node = node_queue.front();
		node_queue.pop();
		if (!node->board.is_terminal()) {
			if (node->board.get_turn() == 1) {
				trainer_x.layer[INPUT_LAYER].enter(node->board.lpos);
				trainer_x.forward_pass();
				cur_Q = max_DQ(node->board.lpos, &trainer_x);
				if (cur_Q > high_Q) high_Q = cur_Q;
			}
			else {
				trainer_o.layer[INPUT_LAYER].enter(node->board.lpos);
				trainer_o.forward_pass();
				cur_Q = min_DQ(node->board.lpos, &trainer_o);
				if (cur_Q < low_Q) low_Q = cur_Q;
			}
			for (int i = 0; i <= node->last_child; i++) {
				if (!crawler.visited_node[node->children[i]->hash]) {
					node_queue.push(node->children[i]);
					crawler.visited_node[node->children[i]->hash] = true;
				}
			}
		}
	}
}

// *********************************************************************************
// **************** Orbit class ****************************************************

Orbit::Orbit (Node* initial_node, Game_tree* tr) {
	tree = tr;
	int hash;
	bool new_hash;
	Node* node = initial_node;
	int first, mid, last;
	for (int s = 0; s < 2; s++) {
		for (int r = 0; r < 4; r++) {
			node = tree->reflect_node( tree->rotate_node_ccw(initial_node, r), s);
			hash = node->board.get_hash();
			new_hash = true;
			first = 0, last = (orbit_hashes.size() - 1), mid = (first + last) / 2;
			while (first <= last) {
				if ( orbit_hashes[mid] == hash) {
					new_hash = false;
					break;
				}
				else if ( orbit_hashes[mid] < hash ) {
					first = mid + 1;
					mid = (first + last) / 2;
				}
				else { 		// orbit_hashes[mid] > hash
					last = mid - 1;
					mid = (first + last) / 2;
				}
			}
			if (new_hash) orbit_hashes.insert(orbit_hashes.begin() + first, hash);
		}
	}
	rep_hash = orbit_hashes[0];
	size = orbit_hashes.size();
}

Orbit::Orbit (int initial_node_hash, Game_tree* tr) : Orbit(tr->table[initial_node_hash], tr) {
}

// *********************************************************************************
// **************** Game tree ******************************************************

Game_tree::Game_tree () {
	hash (&root_node, 0);
	root_node.board.set_depth(0);
	gen_and_link (&root_node);
}

void Game_tree::hash(Node* node, int h) {
	node->hash = h;
	table[h] = node;
	htable[h] = true;
}

void Game_tree::gen_and_link(Node* parent) {
	int child_hash;
	int turn;
	if (!parent->board.is_terminal()) {
		if (!parent->children_were_generated) {
			for (int i = 0; i < CELLS; i++) {
				if ( parent->board.cell_free(i) ) {
					turn = turn_to_ternary(parent->board.get_turn());
					child_hash = (parent->hash) + turn * pow(3,i);
					if (!htable[child_hash]) {
						parent->add_child (child_hash);
						hash( parent->children[parent->last_child], child_hash);
						parent->children[parent->last_child]->board.set_depth( parent->board.get_depth() + 1 );
					}
					else {
						parent->add_child( table[child_hash] );
					}
					parent->children_cells[parent->last_child] = i;
				}
			}
			parent->children_were_generated = true;
		}
		for (int i = 0; i <= parent->last_child; i++) {
			gen_and_link(parent->children[i]);
		}
	}
}

void Game_tree::compute_utilities() {
	root_node.compute_utility();
}

void Game_tree::compute_opti_moves() {
	root_node.compute_opti_move();
}

void Game_tree::reset_Q_values(bool r) {
	root_node.reset_Q_values(r);
}

void Game_tree::reset_double_Q_values(bool r) {
	root_node.reset_double_Q_values(r);
}

Node* Game_tree::rotate_node_ccw (Node* initial_node) {
	int rotated_lpos[CELLS];
	Board rotated_board;
	int rotated_node_hash;
	int rotation[CELLS] = {2, 5, 8, 1, 4, 7, 0, 3, 6};
	for (int i = 0; i < CELLS; i++) {
		rotated_lpos[i] = initial_node->board.lpos[ rotation[i] ];
	}
	rotated_board = Board(rotated_lpos);
	rotated_node_hash = rotated_board.get_hash();

	return table[rotated_node_hash];
}

Node* Game_tree::rotate_node_ccw (Node* initial_node, int n) {
	n = n % 4;
	for (int i = 0; i < n; i++) initial_node = Game_tree::rotate_node_ccw(initial_node);
	return initial_node;
}

Node* Game_tree::reflect_node (Node* initial_node) {
	int reflected_lpos[CELLS];
	Board reflected_board;
	int reflected_node_hash;
	int reflection[CELLS] = {2, 1, 0, 5, 4, 3, 8, 7, 6 };
	for (int i = 0; i < CELLS; i++) {
		reflected_lpos[i] = initial_node->board.lpos[ reflection[i] ];
	}
	reflected_board = Board(reflected_lpos);
	reflected_node_hash = reflected_board.get_hash();
	return table[reflected_node_hash];
}

Node* Game_tree::reflect_node (Node* initial_node, int n) {
	n = n % 2;
	for (int i = 0; i < n; i++) initial_node = Game_tree::reflect_node (initial_node);
	return initial_node;
}


// *********************************************************************************
// *************** Tree crawler ****************************************************

Tree_crawler::Tree_crawler (Game_tree* tr) {
	tree = tr;
}

void Tree_crawler::reset_visits() {
	for (int i = 0; i < BOARDS; i++) visited_node[i] = false;
}


void Tree_crawler::find_orbits() {
	reset_visits();
	find_orbit( &(tree->root_node) );
	cout << "Found " << orbits.size() << " orbits" << endl;
}

void Tree_crawler::find_orbit(Node* node) {
	visited_node[node->hash] = true;
	Orbit node_orbit = Orbit(node, tree);
	orbits.push_back(node_orbit);
	for (int i = 0; i < node_orbit.size; i++) {
		visited_node[ node_orbit.orbit_hashes[i] ] = true;
	}
	for (int i = 0; i <= node->last_child; i++) {
		if ( !visited_node[node->children[i]->hash] )
			find_orbit(node->children[i]);
	}
}

void Tree_crawler::find_orbits_breadthfirst() {
	queue<Node*> nodes_to_visit;
	Node* node = &(tree->root_node);
	Orbit node_orbit = Orbit(node, tree);
	orbits.push_back(node_orbit);
	reset_visits();
	nodes_to_visit.push(node);
	visited_node[0] = true;
	while (nodes_to_visit.size() > 0) {
		node = nodes_to_visit.front();
		nodes_to_visit.pop();
		if (!node->board.is_terminal()) {
			for (int i = 0; i <= node->last_child; i++) {
				if (!visited_node[node->children[i]->hash]) {
					nodes_to_visit.push(node->children[i]);
					node_orbit = Orbit(node->children[i], tree);
					orbits.push_back(node_orbit);
					for (int i = 0; i < node_orbit.size; i++) {
						visited_node[ node_orbit.orbit_hashes[i] ] = true;
					}
				}
			}
		}
	}
	cout << "Found " << orbits.size() << " orbits" << endl;
}


void Tree_crawler::compute_opti_move (Node* node) {
	node->compute_opti_move();
	visited_node[node->hash] = true;
	for (int i = 0; i <= node->last_child; i++) {
		if (!node->children[i]->board.is_terminal()
		and !visited_node[ node->children[i]->hash ])
			compute_opti_move(node->children[i]);
	}
}

void Tree_crawler::compute_opti_moves() {
	reset_visits();
	compute_opti_move (&tree->root_node);
}

void Tree_crawler::compute_utility( Node* node) {
	node->compute_utility();
	visited_node[node->hash] = true;
	for (int c = 0; c <= node->last_child; c++) {
		if (!visited_node[node->children[c]->hash])
			compute_utility(node->children[c]);
	}
}


void Tree_crawler::compute_utilities() {
	reset_visits();
	compute_utility (&tree->root_node);
}

// *********************************************************************************
// ************** Board ************************************************************

Board::Board (int n)  {
	hash = n;
	if (n != 0) {
          int temp;
          for (int i = 0; i < CELLS; i++) {
              temp = n % COLMS;
              switch (temp) {
                  case 1  :  lpos[i] = 1;
                             break;
                  case 2  :  lpos[i] = -1;
                             break;
                  default :  lpos[i] = 0;
                             break;
              }
              n = n/COLMS;
          }
	}
	else fill(begin(lpos), begin(lpos) + CELLS, 0);
	int num_X = count(1), num_O = count(-1);
	turn = 1 - 2*(num_X - num_O);			// 1 if X is to move;   -1 if O is to move
	for (int l = 0; l < LINES; l++) for (int c = 0; c < 3; c++) line[l].entry[c] = lpos[line_cell_to_board_cell(l,c)];
	check_win(1);
	if (winner != 1) check_win(-1);
	int free_spaces = 0;
	for (int i = 0; i < CELLS; i++) if (lpos[i] == 0) free_spaces++;
	if ((winner == 1) or (winner == -1) or (free_spaces == 0)) terminal = true;
	else terminal = false;
	reward = winner;
}

Board::Board (int lpos[]) : Board( lpos_to_hash(lpos) )  {
}

int Board::count(int pl) {
	int num = 0;
	for (int i = 0; i < CELLS; i++) if (lpos[i] == pl) num++;
	return num;
}

void Board::check_win(int pl) {
	for (int i = 0; i < LINES; i++) {
		if (line[i].check_winning_line(pl)) {
			winner = pl;
			break;
			}
		}
}

bool Board::cell_free(int i) {
	if (lpos[i] == 0) return true;
	else return false;
}

int Board::get_depth() {
	return depth;
}

int Board::get_hash() {
	return hash;
}

int Board::get_reward() {
	return reward;
}

int Board::get_turn() {
	return turn;
}

int Board::get_winner() {
	return winner;
}

void Board::set_depth(int n) {
	depth = n;
}

bool Board::is_terminal() {
	return terminal;
}

void Board::print() {
	cout << endl;
	cout << "Hash: " << hash << ";  Terminal: " << terminal << endl;
	cout << "Turn: " << turn << ";  Reward: " << reward << endl;
	cout << lpos[0] << " " << lpos[1] << " " << lpos[2] << endl;
	cout << lpos[3] << " " << lpos[4] << " " << lpos[5] << endl;
	cout << lpos[6] << " " << lpos[7] << " " << lpos[8] << endl;
	cout << endl;
}

int Board::Line::count_marks (int mark) {
	int count = 0;
	for (int i = 0; i < ROWS; i++) if (entry[i] == mark) count++;
	return count;
}

bool Board::Line::spaces_free() {
	for (int i = 0; i < ROWS; i++) if (entry[i] == 0) return true;
	return false;
}

bool Board::Line::check_winning_line (int pl) {
	if (count_marks(pl) == 3) return true;
	else return false;
}

bool Board::Line::check_one_of_three (int pl) {
	if ( (count_marks(pl) == 1) and (count_marks(0) == 2) ) return true;
	else return false;
}


bool Board::Line::check_two_of_three (int pl) {
	if ((count_marks(pl) == 2) and spaces_free()) return true;
	else return false;
}

int Board::Line::find_two_of_three (int pl) {
	if (check_two_of_three(pl)) {
		if (entry[0] == 0) return 0;
		else if (entry[1] == 0) return 1;
		else if (entry[2] == 0) return 2;
	}
	return -1; // Error
}

// *********************************************************************************
// ************ Q trainer **********************************************************

Q_trainer::Q_trainer (float a, float g, float e) {
	ALPHA = initALPHA = a;
	GAMMA = initGAMMA = g;
	EPSILON = initEPSILON = e;
	tree.compute_utilities();
}

Q_trainer::Q_trainer (float a, float g, float e, int epsbdry) {
	EXPLOREBDRY = epsbdry;

	ALPHA = initALPHA = a;
	GAMMA = initGAMMA = g;
	EPSILON = initEPSILON = e;
	tree.compute_utilities();
}

void Q_trainer::train_double (int eps) {
	Node *cur, *mid, *next;
	int move1, move2;
	int coin;
	double cur_alpha;
	random_device rd;
	mt19937 gen(rd());
	uniform_real_distribution<> dis(0.0, 1.0);
	for (int i = 0; i < eps; i++) {
		cur = &(tree.root_node);
		if (dis(gen) < EPSILON) move1 = cur->next_move(RANDOM);
		else move1 =  cur->next_move(DOUBLEQ);
		mid = cur->children[move1];
		cur->refresh_double_Q_data();
		mid->refresh_double_Q_data();
		while (!cur->board.is_terminal()) {
			coin = rand() % 2;
			if (!mid->board.is_terminal()) {
				if (dis(gen) < EPSILON) move2 = move2 = mid->next_move(RANDOM);
				else move2 = mid->next_move(DOUBLEQ);
				next = mid->children[move2];
				next->refresh_double_Q_data();
				switch (cur->board.get_turn()) {
					case 1  : {
	if (coin == 0)
		cur->Q1.X[move1] += ALPHA*(next->board.get_reward() + GAMMA*(next->Q2.X[next->Q1_arg_optimal]) - cur->Q1.X[move1]);
	else
		cur->Q2.X[move1] += ALPHA*(next->board.get_reward() + GAMMA*(next->Q1.X[next->Q2_arg_optimal]) - cur->Q2.X[move1]);
						break;
						}

					case -1 : {
	if (coin == 0)
		cur->Q1.O[move1] += ALPHA*(next->board.get_reward() + GAMMA*(next->Q2.O[next->Q1_arg_optimal]) - cur->Q1.O[move1]);
	else
		cur->Q2.O[move1] += ALPHA*(next->board.get_reward() + GAMMA*(next->Q1.O[next->Q2_arg_optimal]) - cur->Q2.O[move1]);

						break;
						}
				}
			}
			else {
				switch (cur->board.get_turn()) {

					case 1  : {
	if (coin == 0)
		cur->Q1.X[move1] += ALPHA*(mid->board.get_reward() - cur->Q1.X[move1]);
	else
		cur->Q2.X[move1] += ALPHA*(mid->board.get_reward() - cur->Q2.X[move1]);
					break;
						}
					case -1 : {
	if (coin == 0)
		cur->Q1.O[move1] += ALPHA*(mid->board.get_reward() - cur->Q1.O[move1]);
	else
		cur->Q2.O[move1] += ALPHA*(mid->board.get_reward() - cur->Q2.O[move1]);
					break;
						}

				}
			}

			cur = mid;
			mid = next;
			move1 = move2;
		}
		cur_ep++;
		if ( (cur_ep <= EXPLOREBDRY) and ((cur_ep % EXPLOREBDRYSTEP) == 0) ) EPSILON = max(0.1, (EPSILON-0.1));
		if ( (cur_ep % (EPOCHS/100)) == 0 ) ALPHA *= 0.9;
	}
}

void Q_trainer::train (int eps) {
	Node *cur, *mid, *next;
	int move1, move2;
	random_device rd;
	mt19937 gen(rd());
	uniform_real_distribution<> dis(0.0, 1.0);
	for (int i = 0; i < eps; i++) {
		cur = &(tree.root_node);
		if (dis(gen) < EPSILON) move1 = cur->next_move(RANDOM);
		else move1 =  cur->next_move(QTAB);
		mid = cur->children[move1];
		while (!cur->board.is_terminal()) {
			if (!mid->board.is_terminal()) {
				if (dis(gen) < EPSILON) move2 = mid->next_move(RANDOM);
				else move2 = mid->next_move(QTAB);
				next = mid->children[move2];
				next->refresh_Q_data();
				switch (cur->board.get_turn()) {

	case 1  : cur->Q.X[move1] += ALPHA*(next->board.get_reward() + GAMMA*(next->best_Q) - cur->Q.X[move1]);
						break;
	case -1 : cur->Q.O[move1] += ALPHA*(next->board.get_reward() + GAMMA*(next->best_Q) - cur->Q.O[move1]);
						break;

				}
			}
			else {
				switch (cur->board.get_turn()) {

	case 1  : cur->Q.X[move1] += ALPHA*(mid->board.get_reward() - cur->Q.X[move1]);
						break;
	case -1 : cur->Q.O[move1] += ALPHA*(mid->board.get_reward() - cur->Q.O[move1]);
						break;

				}
			}
			cur = mid;
			mid = next;
			move1 = move2;
		}
		cur_ep++;
		if ( (cur_ep <= EXPLOREBDRY) and ((cur_ep % EXPLOREBDRYSTEP) == 0) ) EPSILON = max(0.1, (EPSILON-0.1));
		if ( (cur_ep % (EPOCHS/100)) == 0 ) ALPHA *= 0.9;
	}
}

void Q_trainer::reset_hypers() {
	ALPHA = initALPHA;
	GAMMA = initGAMMA;
	EPSILON = initEPSILON;
}

void Q_trainer::reset_Q() {
	cur_ep = 0;
	tree.reset_Q_values(QRAND);
}

void Q_trainer::reset_double_Q() {
	cur_ep = 0;
	tree.reset_double_Q_values(QRAND);
}

wlt operator+ (wlt left, wlt right) {
	wlt temp;
	temp.w = left.w + right.w;
	temp.l = left.l + right.l;
	temp.t = left.t + right.t;
	for (int i = 0; i <= CELLS; i++) {
		temp.end_depths[i] = left.end_depths[i] + right.end_depths[i];
		temp.win_depths[i] = left.win_depths[i] + right.win_depths[i];
	}
	return temp;
}

float_wlt operator/(float_wlt in, int n) {
	in.w = in.w / n;
	in.l = in.l / n;
	in.t = in.t / n;
	for (int i = 0; i <= CELLS; i++) {
		in.end_depths[i] = in.end_depths[i] / n;
		in.win_depths[i] = in.win_depths[i] / n;
	}
	return in;
}

float_wlt& float_wlt::operator= (const wlt& in) {
	w = (float) in.w;
	l = (float) in.l;
	t = (float) in.t;
	for (int i = 0; i <= CELLS; i++) {
		end_depths[i] = (float) in.end_depths[i];
		win_depths[i] = (float) in.win_depths[i];
	}
	return *this;
}

Q_values operator+(Q_values Q1, Q_values Q2) {
	Q_values Q_sum;
	for (int i = 0; i <= CELLS; i++) {
		Q_sum.X[i] = Q1.X[i] + Q2.X[i];
		Q_sum.O[i] = Q1.O[i] + Q2.O[i];
	}
	return Q_sum;
}

wlt Q_trainer::battle (strategy plstrat, strategy compstrat, bool initrand) {
	wlt ledger;
	Node* battle = &(tree.root_node);
	for (int i = 0; i < NUMBATTLES; i++) {
		battle = &(tree.root_node);
		if (initrand) {
			battle = battle->next_move_node(RANDOM);            // Randomize initial move
			battle = battle->next_move_node(compstrat);
		}
		while ( !battle->board.is_terminal() ) {
			battle = battle->next_move_node(plstrat);
			if (battle->board.is_terminal()) {
				ledger.end_depths[battle->board.get_depth()]++;
				if (battle->board.get_winner() == 1) {
					ledger.w++;
					ledger.win_depths[battle->board.get_depth()]++;
				}
				else
					ledger.t++;
				break;
			}
			else {
				battle = battle->next_move_node(compstrat);
				if (battle->board.is_terminal()) {
					ledger.end_depths[battle->board.get_depth()]++;
					if (battle->board.get_winner() == -1)
						ledger.l++;
					else
						ledger.t++;
					break;
				}
			}
		}
	}
	return ledger;
}

float_wlt Q_trainer::battle (strategy plstrat, strategy compstrat, int numit, bool initrand) {
	wlt total_ledger, cur_game_ledger;

	for (int i = 0; i < numit; i++) {
		cur_game_ledger = battle(plstrat, compstrat, initrand);
		total_ledger = total_ledger + cur_game_ledger;
	}

	float_wlt float_ledger;
	float_ledger = total_ledger;
	float_ledger = (float_ledger / numit) / NUMBATTLES;

	cout << "Averaged over " << numit << " iterations" << endl;
	cout << "W: " << float_ledger.w << endl;
	cout << "L: " << float_ledger.l << endl;
	cout << "T: " << float_ledger.t << endl;
	if (DEPTHVERBOSE) {
		for (int i=5; i<= CELLS; i++) {
			cout << "At depth " << i << ".  Finished: " << float_ledger.end_depths[i] << " ; Won: " << float_ledger.win_depths[i] << endl;
		}
	}
	cout << endl;

	return float_ledger;
}

string Q_trainer::get_plot_data (int epochs, int epochstep, int numit, bool d, bool wrec, bool lrec, bool trec) {
	int points = epochs / epochstep + 1;
	float w[points], l[points], t[points];
	string wRecord = "", lRecord = "", tRecord = "";
	float_wlt data;

	for (int i = 0; i < points; i++) {
		w[i] = l[i] = t[i] = 0.0;
	}

	for (int it = 0; it < numit; it++) {
		if (d) reset_double_Q();
		else reset_Q();

		reset_hypers();

		cout << "**********************************************************" << endl;
		cout << "**********************************************************" << endl;
		cout << "**********************************************************" << endl;
		cout << "EPOCH:" << it << endl;

		for (int ep = 0; ep < points; ep++ ) {
			cout << "Current episode: " << cur_ep << endl;
			if (d) train_double(epochstep);
			else train(epochstep);

			data = battle(COMPSTRAT, COMPOPP, 1);
			if (wrec) w[ep] += data.w;
			if (lrec) l[ep] += data.l;
			if (trec) t[ep] += data.t;
		}
	}
	for (int i = 0; i < points; i++) {
		if (wrec) {	w[i] = w[i] / numit;
				wRecord += (to_string(i*epochstep) + " " + to_string(w[i]) + ","); }
		if (lrec) {	l[i] = l[i] / numit;
				lRecord += (to_string(i*epochstep) + " " + to_string(l[i]) + ","); }
		if (trec) {	t[i] = t[i] / numit;
				tRecord += (to_string(i*epochstep) + " " + to_string(t[i]) + ","); }
	}
	string output_string = "";
		if (wrec) output_string += (wRecord + ";");
		if (lrec) output_string += (lRecord + ";");
		if (trec) output_string += (tRecord + ";");
	return output_string;
}

vector<string> Q_trainer::get_plot_data_w_stdev (int epochs, int epochstep, int numit, bool d, bool wrec, bool lrec, bool trec) {
	int points = epochs / epochstep + 1;
	float w[points], l[points], t[points];
	float wdev[points], ldev[points], tdev[points];
	vector<string> wRecord;
	string wCurrentRecord = "";
	string wAverageRecord = "", lRecord = "", tRecord = "";
	string wAverageRecordPlusDev = "";
	string wAverageRecordMinusDev = "";
	float_wlt data;

	for (int i = 0; i < points; i++) {
		w[i] = l[i] = t[i] = wdev[i] = ldev[i] = tdev[i] = 0.0;
	}
	for (int it = 0; it < numit; it++) {
		if (d) reset_double_Q();
		else reset_Q();
		reset_hypers();
		cout << "**********************************************************" << endl;
		cout << "**********************************************************" << endl;
		cout << "**********************************************************" << endl;
		cout << "EPOCH:" << it << endl;

		for (int ep = 0; ep < points; ep++ ) {
			cout << "Current episode: " << cur_ep << endl;
			if (d) train_double(epochstep);
			else train(epochstep);
			data = battle(COMPSTRAT, COMPOPP, 1);
			if (wrec) { w[ep] += data.w;
				    wdev[ep] += data.w * data.w;
				    wCurrentRecord += (to_string(ep * epochstep) + " " + to_string(data.w) + ","); }
			if (lrec) l[ep] += data.l;
			if (trec) t[ep] += data.t;
		}
		wRecord.push_back(wCurrentRecord);
		wCurrentRecord.clear();
	}
	for (int i = 0; i < points; i++) {
		if (wrec) {	w[i] = w[i] / numit;
				wdev[i] = sqrt( ( wdev[i] / numit ) - w[i]*w[i] ); }
		if (lrec) {	l[i] = l[i] / numit;
				lRecord += (to_string(i*epochstep) + " " + to_string(l[i]) + ","); }
		if (trec) {	t[i] = t[i] / numit;
				tRecord += (to_string(i*epochstep) + " " + to_string(t[i]) + ","); }
	}
	for (int i = 0; i < points; i++) {
		wAverageRecordPlusDev += (to_string(i*epochstep) + " " + to_string(w[i] + wdev[i]) + ",");
		wAverageRecord += (to_string(i*epochstep) + " " + to_string(w[i]) + ",");
		wAverageRecordMinusDev += (to_string(i*epochstep) + " " + to_string(w[i] - wdev[i]) + ",");
	}
	string output_string = "";
		if (wrec) output_string += (wAverageRecordPlusDev + ";" + wAverageRecord + ";" + wAverageRecordMinusDev + ";");
		if (lrec) output_string += (lRecord + ";");
		if (trec) output_string += (tRecord + ";");
	wRecord.push_back(output_string);
	return wRecord;
}

string Q_trainer::get_plot_data_avg_w_stdev (int epochs, int epochstep, int numit, bool d, bool wrec, bool lrec, bool trec) {
	int points = epochs / epochstep + 1;
	float w[points], l[points], t[points];
	float wdev[points], ldev[points], tdev[points];
	string wAverage = "", lAverage = "", tAverage = "",
	       wAveragePlusDev = "", wAverageMinusDev = "",
	       lAveragePlusDev = "", lAverageMinusDev = "",
	       tAveragePlusDev = "", tAverageMinusDev = "";
	float_wlt data;

	for (int i = 0; i < points; i++) {
		w[i] = l[i] = t[i] = wdev[i] = ldev[i] = tdev[i] = 0.0;
	}
	for (int it = 0; it < numit; it++) {
		if (d) reset_double_Q();
		else reset_Q();
		reset_hypers();

		cout << "EPOCH:" << it << endl;

		for (int ep = 0; ep < points; ep++ ) {
			cout << "Current episode: " << cur_ep << endl;
			if (d) train_double(epochstep);
			else train(epochstep);
			data = battle(COMPSTRAT, COMPOPP, 1);
			if (wrec) { w[ep] += data.w;
				    wdev[ep] += data.w * data.w; }
			if (lrec) { l[ep] += data.l;
                                    ldev[ep] += data.l * data.l; }
			if (trec) { t[ep] += data.t;
                                    tdev[ep] += data.t * data.t; }
		}
	}
	for (int i = 0; i < points; i++) {
		if (wrec) {	w[i] = w[i] / numit;
				wdev[i] = sqrt( ( wdev[i] / numit ) - w[i]*w[i] ); } 	// Use sum of squares to compute variance
		if (lrec) {	l[i] = l[i] / numit;
                                ldev[i] = sqrt( ( ldev[i] / numit ) - l[i]*l[i] ); }
		if (trec) {	t[i] = t[i] / numit;
                                tdev[i] = sqrt( ( tdev[i] / numit ) - t[i]*t[i] ); }
	}
        if (wrec) {
   		for (int i = 0; i < points; i++) {
			wAveragePlusDev += (to_string(i*epochstep) + " " + to_string(w[i] + wdev[i]) + ",");
			wAverage += (to_string(i*epochstep) + " " + to_string(w[i]) + ",");
			wAverageMinusDev += (to_string(i*epochstep) + " " + to_string(w[i] - wdev[i]) + ",");
		}
	}
	if (lrec) {
   		for (int i = 0; i < points; i++) {
			lAveragePlusDev += (to_string(i*epochstep) + " " + to_string(l[i] + ldev[i]) + ",");
			lAverage += (to_string(i*epochstep) + " " + to_string(l[i]) + ",");
			lAverageMinusDev += (to_string(i*epochstep) + " " + to_string(l[i] - ldev[i]) + ",");
		}
	}
	if (trec) {
   		for (int i = 0; i < points; i++) {
			tAveragePlusDev += (to_string(i*epochstep) + " " + to_string(t[i] + tdev[i]) + ",");
			tAverage += (to_string(i*epochstep) + " " + to_string(t[i]) + ",");
			tAverageMinusDev += (to_string(i*epochstep) + " " + to_string(t[i] - tdev[i]) + ",");
		}
	}
	string output_string = "";
		if (wrec) output_string += (wAveragePlusDev + ";" + wAverage + ";" + wAverageMinusDev + ";");
		if (lrec) output_string += (lAveragePlusDev + ";" + lAverage + ";" + lAverageMinusDev + ";");
		if (trec) output_string += (tAveragePlusDev + ";" + tAverage + ";" + tAverageMinusDev + ";");
	return output_string;
}


// *********************************************************************************
// ************ Node ***************************************************************

void Node::add_child ( Node* child ) {
	last_child++;
	children[last_child] = child;

}

void Node::add_child (Board child_board ) {
	Node* child = new Node(child_board);
	add_child(child);

}

void Node::add_child (int child_board_index) {
	Board child_board = Board(child_board_index);
	add_child (child_board);

}

Node::Node() {
	board = Board(0);
}

Node::Node(Board init_board) {
	board = init_board;

}

Node::Node(int init_board_index) {
	hash = init_board_index;
	board = Board(init_board_index);
}

int Node::find_child_by_cell (int cell) {
	int first = 0, last = last_child;
	int mid = (first + last) / 2;
	while(first <= last) {
		if (children_cells[mid] == cell) return mid;
		else if (children_cells[mid] < cell) {
			first = mid + 1;
			mid = (first + last) / 2;
		}
		else { 	// children_cells[mid] > cell
			last = mid - 1;
			mid = (first + last) / 2;
		}
	}
	return -1;
}

// Optibot information
const int CORNER[4] = {0, 2, 6, 8};					// Corner cells
const int OPPOSITECORNER[4] = {8, 6, 2, 0};				// Opposite corners from corner cells
const int SIDE[4] = {1, 3, 5, 7};					// Side cells
const int CENTER = 4; 							// The center
const vector<int> LINESTHROUGHCELL[9] =       { {0, 3, 6}, 		//
						{1, 3}, 		// Ids of lines passing through each
						{2, 3, 7}, 		// of the board cells
						{0, 4},			//
						{1, 4, 6, 7},
						{2, 4},
						{0, 5, 7},
						{1, 5},
						{2, 5, 6} };

int Node::find_opti_move() {
	// Follows strategy of CROWLEY and SIEGLER  (slightly modified)

	int n = 0, lc = 0;
	bool cell_recommended[CELLS] = {false};

	// Win
	for (int i = 0; i < LINES; i++) {
		if ( board.line[i].check_two_of_three(board.get_turn())) {
			n = board.line[i].find_two_of_three(board.get_turn());
			n = line_cell_to_board_cell(i, n);
			if (!cell_recommended[n]) {
				cell_recommended[n] = true;
				opti_moves.push_back( n );
			}
		}
	}
	if (!opti_moves.empty()) {
		opti_strat = WIN;
		return opti_moves[0];
	}

	// Block opponent's win
	for (int i = 0; i < LINES; i++) {
		if ( board.line[i].check_two_of_three(-board.get_turn())) {
			n = board.line[i].find_two_of_three(-board.get_turn());
			n = line_cell_to_board_cell(i, n);
			if (!cell_recommended[n]) {
				cell_recommended[n] = true;
				opti_moves.push_back( n );
			}
		}
	}
	if (!opti_moves.empty()) {
		opti_strat = BLOCK_WIN;
		return opti_moves[0];
	}

	// Fork
	for (int i = 0; i < CELLS; i++) {
		if (board.lpos[i] == 0) {
			for (int a = 0; a < LINESTHROUGHCELL[i].size(); a++) {
			for (int b = a + 1; b < LINESTHROUGHCELL[i].size(); b++) {
				if ( board.line[LINESTHROUGHCELL[i][a]].check_one_of_three(board.get_turn()) and
				     board.line[LINESTHROUGHCELL[i][b]].check_one_of_three(board.get_turn()) )
					{ 	opti_strat = FORK;
						opti_moves.push_back(i);
					}
			}
			}
		}
	}
	if (!opti_moves.empty()) return opti_moves[0];

	// Block potential forks or force an advantegous block
	Node* future_node;
	Board::Line temp_line;

	bool forced_block_is_bad, forced_move, win_next_turn;
	int num_opp_forks;
	bool winning_force_block[CELLS] = {false, false, false, false, false, false, false, false, false};

	for (int i = 0; i < CELLS; i++) {
		if ( board.lpos[i] == 0 ) {
			for (int line = 0; line < LINESTHROUGHCELL[i].size(); line++) {
				if ( board.line[LINESTHROUGHCELL[i][line]].check_one_of_three(board.get_turn()) ) {
					forced_block_is_bad = win_next_turn = false;

					// Check if the forced block is a fork or a loss
					forced_move = true;
					future_node = this;
					n = i;

					while (forced_move) {	// Go through the sequence of forced moves
						future_node = future_node->children[future_node->find_child_by_cell(n)];
						if (future_node->board.is_terminal()) {
							forced_move = false;
							switch(board.get_turn() * future_node->board.get_winner()) {

								case -1		          :  forced_block_is_bad = true;
									     		     break;
								case 1		          :  forced_block_is_bad = false;
											     winning_force_block[i] = true;
									      		     break;
							}
						}
						else {
							for (int l = 0; l < LINES; l++) {
								if (future_node->board.line[l].check_two_of_three(future_node->board.get_turn())) {
									forced_move = false;
									win_next_turn = true;
									switch (board.get_turn() * future_node->board.get_turn()) {
										case -1   :  forced_block_is_bad = true;
											     break;
										case 1	  :  forced_block_is_bad = false;
											     break;
									}
								}
							}
							if (!win_next_turn) {
							forced_move = false;
							for (int a = 0; a < LINESTHROUGHCELL[n].size(); a++) {
								temp_line = future_node->board.line[ LINESTHROUGHCELL[n][a] ];
								if (temp_line.check_two_of_three (-future_node->board.get_turn()) ) {
									lc = temp_line.find_two_of_three (-future_node->board.get_turn() );
									n = line_cell_to_board_cell (LINESTHROUGHCELL[n][a], lc);
									forced_move = true;
								}
							}
							}
						}
					}
					// Opponent should not end up with a potential fork on their turn

					if (!future_node->board.is_terminal()) {
						num_opp_forks = 0;
						for (int cell = 0; cell < CELLS; cell++) {
							if (future_node->board.lpos[cell] == 0) {
								for (int a = 0; a < LINESTHROUGHCELL[cell].size(); a++) {
								for (int b = a+1; b < LINESTHROUGHCELL[cell].size(); b++) {
									if (future_node->board.line[ LINESTHROUGHCELL[cell][a] ].check_one_of_three(-board.get_turn()) and
									    future_node->board.line[ LINESTHROUGHCELL[cell][b] ].check_one_of_three(-board.get_turn()) and
										!cell_recommended[cell]) {
											num_opp_forks++;
											cell_recommended[cell] = true;
									}
								}
								}
							}
						}
						switch (future_node->board.get_turn() * board.get_turn()) {
							case -1   :   if (num_opp_forks > 0) forced_block_is_bad = true;
							case 1    :   if (num_opp_forks > 1) forced_block_is_bad = true;
						}
					}
					if (!forced_block_is_bad) {
						opti_strat = FORCE_BLOCK;
						opti_moves.push_back(i);
					}

				}	// end of if
			} 		// end of checking through lines
		} 							// end of if(lpos[i] == 0)
	}								// end of looping through i

	if (!opti_moves.empty()) {
		for (int i = 0; i < CELLS; i++) {
			if (winning_force_block[i]) return i;
		}
		return opti_moves[0];
	}

	// Center
	if (board.lpos[CENTER] == 0) {
		opti_strat = CENTER_MOVE;
		opti_moves.push_back(CENTER);
	}
	if (!opti_moves.empty()) return opti_moves[0];

	// Opposite corner
	for (int i = 0; i < 4; i++) {
		if ( (board.lpos[CORNER[i]] == -board.get_turn()) and (board.lpos[OPPOSITECORNER[i]] == 0) ) {
			opti_strat = OPP_CORNER;
			opti_moves.push_back( OPPOSITECORNER[i] );

		}
	}
	if (!opti_moves.empty()) return opti_moves[0];

	// Empty corner
	for (int i = 0; i < 4; i++) {
		if ( board.lpos[CORNER[i]] == 0 ) {
			opti_strat = CORNER_MOVE;
			opti_moves.push_back( CORNER[i] );
		}
	}
	if (!opti_moves.empty()) return opti_moves[0];

	// Empty side
	for (int i = 0; i < 4; i++) {
		if ( board.lpos[SIDE[i]] == 0 ) {
			opti_strat = SIDE_MOVE;
			opti_moves.push_back( SIDE[i] );
		}
	}
	if (!opti_moves.empty()) return opti_moves[0];

	opti_strat = NO_MOVE;
	return -1; // Error
}

void Node::compute_opti_move() {
	opti_move = find_child_by_cell ( find_opti_move() );
	sort(opti_moves.begin(), opti_moves.end());
}

void Node::compute_utility() {
	if (board.is_terminal())
		utility = board.get_winner();
	else {
		// Compute utility with alpha-beta pruning
	Node* child;

	switch (board.get_turn()) {
	case 1  :	{   utility = -INFTY;
			for (int i = 0; i <= last_child; i++) {
				child = children[i];
				child->compute_utility();
				if (child->utility > utility) {
					utility = child->utility;
					minimax_move = i;
				}
				//child->alpha = utility;
				//if (utility >= beta)
				//	break;
			}
			break;
			}


	case -1 :	{   utility = INFTY;
			for (int i = 0; i <= last_child; i++) {
				child = children[i];
				child->compute_utility();
				if (child->utility < utility) {
					utility = child->utility;
					minimax_move = i;
				}
				//child->beta = utility;
				//if (utility <= alpha)
				//	break;
			}
			break;
			}
		}
	}
}

void Node::refresh_Q_data () {
	best_Q = 0;
	if (!board.is_terminal()) {
		switch (board.get_turn()) {
			case 1  : {
				best_Q = -INFTY;
				for (int i = 0; i <= last_child; i++) {
					if (Q.X[i] > best_Q) {
						best_Q = Q.X[i];
						TQ_move = i;
					}
				}
				break;
				}
			case -1 : {
				best_Q = INFTY;

				for (int i = 0; i <= last_child; i++) {
					if (Q.O[i] < best_Q) {
						best_Q = Q.O[i];
						TQ_move = i;
					}
				}
				break;
			}
		}
	}
}

void Node::reset_Q_values(bool r) {
	Q.reset(r);
	for (int i = 0; i <= last_child; i++) children[i]->reset_Q_values(r);
}

void Node::refresh_double_Q_data () {
	Q_sum = Q1 + Q2;
	if (!board.is_terminal()) {
		int turn = board.get_turn();
		Q1_arg_optimal = find_Q_arg_optimal (&Q1, turn);
		Q2_arg_optimal = find_Q_arg_optimal (&Q2, turn);
		double_TQ_move = find_Q_arg_optimal (&Q_sum, turn);
	}
}


int Node::find_Q_arg_optimal (Q_values* Q_pointer, int turn) {
	double Q_optimal;
	int arg_optimal = -1;
	Q_values Q = *Q_pointer;
	switch (turn) {
		case 1    : 	Q_optimal = -INFTY;
				for (int i = 0; i <= last_child; i++) {
					if (Q.X[i] > Q_optimal) {
						Q_optimal = Q.X[i];
						arg_optimal = i;
					}
			      	}
				break;
		case -1   :	Q_optimal = INFTY;
				for (int i = 0; i <= last_child; i++) {
					if (Q.O[i] < Q_optimal) {
						Q_optimal = Q.O[i];
						arg_optimal = i;
					}
				}
				break;

	}
	return arg_optimal;
}

void Node::reset_double_Q_values(bool r) {
	Q1.reset(r);
	Q2.reset(r);
	Q_sum.reset(r);
	for (int i =0; i <= last_child; i++) children[i]->reset_double_Q_values(r);
}

Q_values::Q_values (bool r) {
	reset(r);
}

void Q_values::reset(bool r) {
	if (r) {	// Seed initial Q-table with small random numbers
		for (int i = 0; i < CELLS; i++) {
			X[i] = ((float) (rand() % 2000) / 20000) - 0.05;
			O[i] = ((float) (rand() % 2000) / 20000) - 0.05;
		}
	}
	else {
		for (int i = 0; i < CELLS; i++) {
			X[i] = 0;
			O[i] = 0;
		}
	}
}

int Node::next_move(strategy strat) {
	switch (strat) {
		case MINIMAX   : return minimax_move;
		case RANDOM    : return rand() % (last_child + 1);
		case HEURIBOT  : return opti_move;
		case QTAB      : {refresh_Q_data(); return TQ_move;}
		case DOUBLEQ   : {refresh_double_Q_data(); return double_TQ_move;}
		case DQN       : return DQ_move;
		default        : {cout << "ERROR: Strategy not recognized in next_move" << endl; return -1;} // Error
	}
}

Node* Node::next_move_node (strategy strat) {
	return children[next_move(strat)];
}

int turn_to_ternary (int turn) {
	switch(turn) {
		case 1  : 	return 1;
		case -1 :	return 2;
		default :	return 0;
	}
}

int line_cell_to_board_cell (int line, int line_cell) {
	switch (line) {
		case 0 : 	if (line_cell == 0) return 0;
				if (line_cell == 1) return 3;
				if (line_cell == 2) return 6;
				break;
		case 1 :	if (line_cell == 0) return 1;
				if (line_cell == 1) return 4;
				if (line_cell == 2) return 7;
				break;
		case 2 :	if (line_cell == 0) return 2;
				if (line_cell == 1) return 5;
				if (line_cell == 2) return 8;
				break;
		case 3 :	if (line_cell == 0) return 0;
				if (line_cell == 1) return 1;
				if (line_cell == 2) return 2;
				break;
		case 4 :	if (line_cell == 0) return 3;
				if (line_cell == 1) return 4;
				if (line_cell == 2) return 5;
				break;
		case 5 :	if (line_cell == 0) return 6;
				if (line_cell == 1) return 7;
				if (line_cell == 2) return 8;
				break;
		case 6 :	if (line_cell == 0) return 0;
				if (line_cell == 1) return 4;
				if (line_cell == 2) return 8;
				break;
		case 7 :	if (line_cell == 0) return 6;
				if (line_cell == 1) return 4;
				if (line_cell == 2) return 2;
				break;
	}
	return -1; // Error
}

int board_cell_to_line_cell (int cell, int line) {
	switch(line) {
		case 0 	:	if (cell == 0) return 0;
				if (cell == 3) return 1;
				if (cell == 6) return 2;
				break;
		case 1  :	if (cell == 1) return 0;
				if (cell == 4) return 1;
				if (cell == 7) return 2;
				break;
		case 2 	:	if (cell == 2) return 0;
				if (cell == 5) return 1;
				if (cell == 8) return 2;
				break;
		case 3  :	if (cell == 0) return 0;
				if (cell == 1) return 1;
				if (cell == 2) return 2;
				break;
		case 4 	:	if (cell == 3) return 0;
				if (cell == 4) return 1;
				if (cell == 5) return 2;
				break;
		case 5  :	if (cell == 6) return 0;
				if (cell == 7) return 1;
				if (cell == 8) return 0;
				break;
		case 6 	:	if (cell == 0) return 0;
				if (cell == 4) return 1;
				if (cell == 8) return 2;
				break;
		case 7  :	if (cell == 6) return 0;
				if (cell == 4) return 1;
				if (cell == 2) return 3;
				break;
	}
	return -1; // Error
}

int lpos_to_hash (int lpos[]) {
	int hash = 0;
	for (int i = 0; i < CELLS; i++) {
		lpos[i] = turn_to_ternary(lpos[i]);
		hash += lpos[i] * pow(3,i);
	}
	return hash;
}

// *********************************************************************************
// ************ Exporter ***********************************************************

void Exporter::output_html_strategy_table (char file_name[], bool breadth) {
	Node* node;
	Board board;
	string turn;
	string winner;
	string opti_class;
	string opti_moves;
	double Q_values[CELLS];
	double DQ_values[CELLS];
	ostringstream terminal_positions;
	ofstream to (file_name);

	Q.train_double(EPOCHS);
        breadth ? crawler.find_orbits_breadthfirst() : crawler.find_orbits();
	crawler.compute_opti_moves();

	DQN_trainer_with_two_nets tr(&Q.tree);
	ifstream weights("weights1280.txt");
	tr.load_weights(weights);
	weights.close();
	tr.store_DQ_actions(&Q.tree);
	tr.find_Q_bounds();
	cout << "High DQ: " << tr.high_Q << endl;
	cout << "Low DQ: " << tr.low_Q << endl;

	to << "<!DOCTYPE html>" << endl;
	to << "<html>" << endl;

	to << "<head>" << endl;
	to << "<link rel=\"stylesheet\" type=\"text/css\" href=\"output-table-style.css\">" << endl;
	to << "<style> </style>" << endl;
	to << "</head>" << endl;
	to << "<body>" << endl;

	if (breadth) to << "<h1> Breadth-first pass of Tic-Tac-Toe game tree </h1>" << endl;
	else to << "<h1> Depth-first pass of Tic-Tac-Toe game tree </h1>" << endl;
	to << "<p> << Back to <a href=\"http://iliasmirnov.com/ttt/\">the project page</a> </p>" << endl;
	to << "<p>The board cells are enumerated as follows:</p>" << endl;
	to << "<table id=\"board\">" << endl;
		to << "<tr>" << endl;
			to << "<td> 0 </td> <td> 1 </td> <td> 2 </td>" << endl;
		to << "</tr> <tr>" << endl;
			to << "<td> 3 </td> <td> 4 </td> <td> 5 </td>" << endl;
		to << "</tr> <tr>" << endl;
			to << "<td> 6 </td> <td> 7 </td> <td> 8 </td>" << endl;
		to << "</tr>" << endl;
	to << "</table>" << endl;
	to << "<p> Red cells indicate the moves judged poor (leading to a loss) by the agent; green cells those judged"
	   << " good (leading to a win); yellow cells those leading to a tie. The remaining colors are interpolated between these three."
           << " Colors of the DQN evaluations are based on the relative position of the DQN value relative to the highest and lowest"
	   << " DQN values, which are " << tr.high_Q << " and " << tr.low_Q << ", respectively. </p>" << endl;
	to << "<h2> Non-terminal positions </h2>" << endl;
	to << "<table id=\"row-of-boards\">" << endl;
	to << "<tr>" << endl;
		to << "<td> Orbit id. </td>" << endl;
		to << "<td> Orbit representative </td>" << endl;
		to << "<td> Whose <br> move? </td>" << endl;
		to << "<td> Optibot classification <br> of best moves <br> (Optibot-approved moves) </td>" << endl;
		to << "<td width=\"130\"> Minimax utilities</td>" << endl;
		to << "<td width=\"130\"> <i>Q</i> values </td>" << endl;
		to << "<td width=\"130\"> DQN values </td>" << endl;
		to << "<td> Other boards in the orbit </td>" << endl;
	to << "</tr>" << endl;
	for (int i = 0; i < crawler.orbits.size(); i++) {
		node = crawler.tree->table[ crawler.orbits[i].orbit_hashes[0] ];
		board = node->board;

		if (!board.is_terminal()) {
			to << "<tr>" << endl;

			// Orbit id.
			to << "<td>" + to_string(i+1) + ". </td>" << endl;

			// Orbit representative diagram
			to << "<td>" << endl;
				print_board_to_html(board, to);
			to << "</td>" << endl;
			// Whose move?
			switch (board.get_turn()) {
				case 1  : turn = "X";
					  break;
				case -1 : turn = "O";
					  break;
			}
			to << "<td><i>" + turn  + "</i></td>" << endl;
			// Rulesbot best move
			switch (node->opti_strat) {
				case WIN 	 : 	opti_class = "Win";
					   		break;
				case BLOCK_WIN   :	opti_class = "Block opponent's win";
							break;
				case FORK 	 : 	opti_class = "Set up fork";
							break;
				case FORCE_BLOCK :	opti_class = "Force opponent to block";
							break;
				case BLOCK_FORK  :	opti_class = "Block opponent's fork";
							break;
				case CENTER_MOVE : 	opti_class = "Move in center";
							break;
				case OPP_CORNER  : 	opti_class = "Move in opposite corner";
							break;
				case CORNER_MOVE : 	opti_class = "Move in corner";
							break;
				case SIDE_MOVE	 : 	opti_class = "Move to side";
							break;
				case NO_MOVE	 : 	opti_class = "No good move (error)";
							break;
			}
			// Best moves according to optibot
			opti_moves = "";
			for (vector<int>::iterator j = node->opti_moves.begin(); j != node->opti_moves.end(); ++j) {
				opti_moves += (" " + to_string(*j) + ";");
			}
			opti_moves.pop_back();
			opti_moves.erase(0,1);
			to << "<td>" + opti_class + "<br> (Moves: " + opti_moves + ") </td>" << endl;
			// Minimax move
				// Fill in utilities of open moves
			for (int i = 0; i < CELLS; i++)
				board.lpos[i] = INFTY;
			for (int i = 0; i <= node->last_child; i++)
				board.lpos[ node->children_cells[i] ] = node->children[i]->utility;
			to << "<td>" << endl;
				print_minimax_board_to_html(board, to, false);
			to << "</td>" << endl;

			//to << "<td>" + to_string( node->children_cells[node->minimax_move] ) + "</td>" << endl;

			// Q-move
			node->refresh_double_Q_data();
				// Fill in Q-values
			for (int i = 0; i < CELLS; i++)
				Q_values[i] = INFTY;
			for (int i = 0; i <= node->last_child; i++) {
				switch (board.get_turn()) {
					case 1   :  Q_values [ node->children_cells[i] ] = ((node->Q1.X[i] + node->Q2.X[i])/2);
						    break;
					case -1  :  Q_values [ node->children_cells[i] ] = ((node->Q1.O[i] + node->Q2.O[i])/2);
						    break;
				}
			}
			to << "<td>" << endl;
				print_Q_board_to_html(Q_values, board.get_turn(), to, false);
			to << "</td>" << endl;

			// DQN-move
			to << "<td>" << endl;
				for (int i = 0; i < CELLS; i++) DQ_values[i] = INFTY;
				for (int i = 0; i <= node->last_child; i++) DQ_values[node->children_cells[i]] = node->DQ_values[node->children_cells[i]];
				print_DQN_board_to_html(DQ_values, board.get_turn(), to, tr.low_Q, tr.high_Q - tr.low_Q, false);
			to << "</td>" << endl;

			//to << "<td>" + to_string( node->children_cells[node->double_TQ_move] ) + "</td>" << endl;

			// Diagrams of other boards in the orbit
			to << "<td id=\"other-boards\">" << endl;
			to << "<div id=\"other-positions\">" << endl;
			for (int j = 1; j < crawler.orbits[i].orbit_hashes.size(); j++) {
				board = crawler.tree->table[ crawler.orbits[i].orbit_hashes[j] ]->board;
				print_board_to_html(board, to, true);
				if (j == 4) to << "<br>" << endl;
			}
			to << "</div>" << endl;
			to << "</td>" << endl;
			to << "</tr>" << endl;
			to << "<tr style=\"border: 0px\"> <td style=\"border : 0px\"> &nbsp </td> </tr>" << endl;
		}

		else 	{	// Position is terminal
			terminal_positions << "<tr>" << endl;
			terminal_positions << "<td>" << to_string(i+1) << ". </td>" << endl;
			terminal_positions << "<td>" << endl;
				print_board_to_string(board, terminal_positions);
			terminal_positions << "</td>" << endl;
			switch(board.get_winner()) {
				case 1 	:	winner = "<i>X</i> victory";
						break;
				case -1 : 	winner = "<i>O</i> victory";
						break;
				case 0  : 	winner = "Tie game";
						break;
			}
			terminal_positions << "<td>" << winner << "</td>" << endl;
			terminal_positions << "<td id=\"other-boards\">" << endl;
			terminal_positions << "<div id=\"other-positions\">" << endl;
			for (int j = 1; j < crawler.orbits[i].orbit_hashes.size(); j++) {
				board = crawler.tree->table[ crawler.orbits[i].orbit_hashes[j] ]->board;
				print_board_to_string (board, terminal_positions, true);
				if (j == 4) terminal_positions << "<br>" << endl;
			}
			terminal_positions << "</div>" << endl;
			terminal_positions << "</td>" << endl;
			terminal_positions << "</tr>" << endl;
			}
	}
	to << "</table>" << endl;


	to << "<h2> Terminal positions </h2>" << endl;
	to << "<table id=\"row-of-boards\">" << endl;
	to << "<tr>" << endl;
		to << "<td> Orbit id. </td>" << endl;
		to << "<td> Orbit representative </td>" << endl;
		to << "<td> Result </td>" << endl;
		to << "<td> Other boards in the orbit </td>" << endl;
	to << "</tr>" << endl;
	to << terminal_positions.str();
	to << "</table>" << endl;
	to << "</body>" << endl;
	to << "</html>" << endl;
	to.close();
}

void Exporter::print_board_to_html(Board board, ofstream& to, bool small) {
	string lpos[CELLS];
		for (int n = 0; n < CELLS; n++) {
			switch(board.lpos[n]) {
				case 1  :  lpos[n] = "X";
					   break;
				case -1 :  lpos[n] = "O";
					   break;
				default :  lpos[n] = "&nbsp";
					   break;
			}
		}

	if (small) to << "<table id=\"board-small\">" << endl;
	else	   to << "<table id=\"board\">" << endl;
		to << "<tr> <td>" + lpos[0] + "</td><td>" + lpos[1] + "</td><td>" + lpos[2] + "</td></tr>" << endl;
		to << "<tr> <td>" + lpos[3] + "</td><td>" + lpos[4] + "</td><td>" + lpos[5] + "</td></tr>" << endl;
		to << "<tr> <td>" + lpos[6] + "</td><td>" + lpos[7] + "</td><td>" + lpos[8] + "</td></tr>" << endl;
		to << "<caption><div id=\"caption\"> H: " + to_string(board.get_hash()) + ".</div></caption>" << endl;
		to << "</table>" << endl;
}

void Exporter::print_minimax_board_to_html(Board board, ofstream& to, bool small) {
	string red = "rgb(255, 0, 0)";
	string green = "rgb(0, 153, 0)";
	string yellow = "rgb(255, 204, 0)";
	string grey = "rgb(240, 240, 240)";
	string x_color, o_color;
	switch (board.get_turn()) {
		case 1  : x_color = green;
			  o_color = red;
			  break;
		case -1 : x_color = red;
			  o_color = green;
			  break;
	}
	string x_cell = "<td style = \"background : " + x_color + "\">";
	string o_cell = "<td style = \"background : " + o_color + "\">";
	string tie_cell = "<td style = \"background : " + yellow  + "\">";
	string used_cell = "<td style = \"background : " + grey + "\">";

	string lpos[CELLS];
		for (int n = 0; n < CELLS; n++) {
			switch(board.lpos[n]) {
				case 1  :  lpos[n] = x_cell + "1</td>";
					   break;
				case -1 :  lpos[n] = o_cell + "-1</td>";
					   break;
				case 0  :  lpos[n] = tie_cell + "0</td>";
					   break;
				default :  lpos[n] = used_cell + "</td>";
					   break;
			}
		}

	if (small) to << "<table id=\"board-small\">" << endl;
	else	   to << "<table id=\"board\" style=\"font-size : 10pt\">" << endl;
		to << "<tr>" + lpos[0] + lpos[1] + lpos[2] + "</tr>" << endl;
		to << "<tr>" + lpos[3] + lpos[4] + lpos[5] + "</tr>" << endl;
		to << "<tr>" + lpos[6] + lpos[7] + lpos[8] + "</tr>" << endl;
		to << "</table>" << endl;
}

void Exporter::print_Q_board_to_html(double Q_values[], int turn, ofstream& to, bool small) {
	string red[10];
	string green[10];

	for (int i = 0; i < 10; i++) {
		red[i] = "rgb(255," + to_string( 204 - 20*(i+1)) + ",0)";
		green[i] = "rgb(" + to_string( 255 - 25*(i+1)) + "," + to_string( 204 - 5*(i+1)) + ",0)";
	}
	string yellow = "rgb(255, 204, 0)";
	string grey = "rgb(240, 240, 240)";

	int Q_value_color_level[CELLS];
	string x_color[CELLS], o_color[CELLS];
	string x_cell[CELLS], o_cell[CELLS];
	string tie_cell = "<td style = \"background : " + yellow  + "\">";
	string used_cell = "<td style = \"background : " + grey + "\">";

	for (int i = 0; i < CELLS; i++) {
		Q_value_color_level[i] = abs(roundf( Q_values[i] * 10 )) - 1;
		Q_value_color_level[i] = max(0, Q_value_color_level[i]);
		Q_value_color_level[i] = min(9, Q_value_color_level[i]);
		switch (turn) {
			case 1  : x_color[i] = green[ Q_value_color_level[i] ];
				  o_color[i] = red[ Q_value_color_level[i] ];
				  break;
			case -1 : x_color[i] = red[ Q_value_color_level[i] ];
				  o_color[i] = green[ Q_value_color_level[i] ];
				  break;
		}
		x_cell[i] = "<td style = \"background : " + x_color[i] + "\">";
		o_cell[i] = "<td style = \"background : " + o_color[i] + "\">";

	}
	int lead = 0;
	ostringstream stream;
	string lpos[CELLS];
		for (int n = 0; n < CELLS; n++) {
			if (0 < Q_values[n] and Q_values[n] < INFTY) lead = 1;
			else if (Q_values[n] == 0) lead = 0;
			else if (Q_values[n] < 0) lead = -1;
			else lead = INFTY;
			stream << fixed << setprecision(1) << Q_values[n];
			lpos[n] = stream.str();
			stream.str("");
			switch(lead) {
				case 1  :  lpos[n] = x_cell[n] + lpos[n] + "</td>";
					   break;
				case -1 :  lpos[n] = o_cell[n] + lpos[n] + "</td>";
					   break;
				case 0  :  lpos[n] = tie_cell + "0.0</td>";
					   break;
				default :  lpos[n] = used_cell + "</td>";
					   break;
			}
		}
	if (small) to << "<table id=\"board-small\">" << endl;
	else	   to << "<table id=\"board\" style=\"font-size : 10pt\">" << endl;
		to << "<tr>" + lpos[0] + lpos[1] + lpos[2] + "</tr>" << endl;
		to << "<tr>" + lpos[3] + lpos[4] + lpos[5] + "</tr>" << endl;
		to << "<tr>" + lpos[6] + lpos[7] + lpos[8] + "</tr>" << endl;
		to << "</table>" << endl;
}

void Exporter::print_DQN_board_to_html(double DQ_values[], int turn, ofstream& to, double DQ_low, double DQ_range, bool small) {
	string red[10];
	string green[10];

	for (int i = 0; i < 10; i++) {
		red[i] = "rgb(255," + to_string( 204 - 20*(i+1)) + ",0)";
		green[i] = "rgb(" + to_string( 255 - 25*(i+1)) + "," + to_string( 204 - 5*(i+1)) + ",0)";
	}
	string yellow = "rgb(255, 204, 0)";
	string grey = "rgb(240, 240, 240)";

	int DQ_value_color_level[CELLS];
	string x_color[CELLS], o_color[CELLS];
	string x_cell[CELLS], o_cell[CELLS];
	string tie_cell = "<td style = \"background : " + yellow  + "\">";
	string used_cell = "<td style = \"background : " + grey + "\">";

	for (int i = 0; i < CELLS; i++) {
		DQ_value_color_level[i] = abs(roundf( (DQ_values[i]-DQ_low)/DQ_range * 10 )) - 1;
		DQ_value_color_level[i] = max(0, DQ_value_color_level[i]);
		DQ_value_color_level[i] = min(9, DQ_value_color_level[i]);
		switch (turn) {
			case 1  : x_color[i] = green[ DQ_value_color_level[i] ];
				  o_color[i] = red[ DQ_value_color_level[i] ];
				  break;
			case -1 : x_color[i] = red[ DQ_value_color_level[i] ];
				  o_color[i] = green[ DQ_value_color_level[i] ];
				  break;
		}
		x_cell[i] = "<td style = \"background : " + x_color[i] + "\">";
		o_cell[i] = "<td style = \"background : " + o_color[i] + "\">";

	}
	int lead = 0;
	ostringstream stream;
	string lpos[CELLS];
		for (int n = 0; n < CELLS; n++) {
			if (0 < DQ_values[n] and DQ_values[n] < INFTY) lead = 1;
			else if (DQ_values[n] == 0) lead = 0;
			else if (DQ_values[n] < 0) lead = -1;
			else lead = INFTY;
			stream << fixed << setprecision(1) << DQ_values[n];
			lpos[n] = stream.str();
			stream.str("");
			switch(lead) {
				case 1  :  lpos[n] = x_cell[n] + lpos[n] + "</td>";
					   break;
				case -1 :  lpos[n] = o_cell[n] + lpos[n] + "</td>";
					   break;
				case 0  :  lpos[n] = tie_cell + "0.0</td>";
					   break;
				default :  lpos[n] = used_cell + "</td>";
					   break;
			}
		}
	if (small) to << "<table id=\"board-small\">" << endl;
	else	   to << "<table id=\"board\" style=\"font-size : 10pt\">" << endl;
		to << "<tr>" + lpos[0] + lpos[1] + lpos[2] + "</tr>" << endl;
		to << "<tr>" + lpos[3] + lpos[4] + lpos[5] + "</tr>" << endl;
		to << "<tr>" + lpos[6] + lpos[7] + lpos[8] + "</tr>" << endl;
		to << "</table>" << endl;
}

void Exporter::print_board_to_string(Board board, ostringstream& tp, bool small) {
	string lpos[CELLS];
		for (int n = 0; n < CELLS; n++) {
			switch(board.lpos[n]) {
				case 1  :  lpos[n] = "X";
					   break;
				case -1 :  lpos[n] = "O";
					   break;
				default :  lpos[n] = "&nbsp";
					   break;
			}
		}

	if (small) tp << "<table id=\"board-small\">" << endl;
	else	   tp << "<table id=\"board\">" << endl;
		tp << "<tr> <td>" + lpos[0] + "</td><td>" + lpos[1] + "</td><td>" + lpos[2] + "</td></tr>" << endl;
		tp << "<tr> <td>" + lpos[3] + "</td><td>" + lpos[4] + "</td><td>" + lpos[5] + "</td></tr>" << endl;
		tp << "<tr> <td>" + lpos[6] + "</td><td>" + lpos[7] + "</td><td>" + lpos[8] + "</td></tr>" << endl;
		tp << "<caption><div id=\"caption\"> H: " + to_string(board.get_hash()) + ".</div></caption>" << endl;
		tp << "</table>" << endl;
}

void fill_strat(Game_tree& tree, int* strat_array, strategy STRAT) {
    Node *cur_node, *child_node;
    queue<Node*> node_queue;
    node_queue.push(&tree.root_node);
    int move;
    bool marked_to_visit[BOARDS];
    for (int i = 0; i < BOARDS; i++) marked_to_visit[i] = false;

    while (node_queue.size() > 0) {
        cur_node = node_queue.front();
        node_queue.pop();
        switch(STRAT) {
            case MINIMAX    :    move = cur_node->minimax_move;
                                 break;
            case HEURIBOT   :    move = cur_node->opti_move;
                                 break;
	    case QTAB       :    move = cur_node->TQ_move;
				 break;
            case DOUBLEQ    :    move = cur_node->double_TQ_move;
                                 break;
            case DQN        :    move = cur_node->DQ_move;
                                 break;
            default         :    move = rand() % (cur_node->last_child + 1);
                                 break;
        }
        strat_array[cur_node->hash] = cur_node->children_cells[move];

        for (int i = 0; i <= cur_node->last_child; i++) {
            child_node = cur_node->children[i];
            if (!child_node->board.is_terminal() && !marked_to_visit[child_node->hash])
                node_queue.push(child_node);
                marked_to_visit[child_node->hash] = true;
        }
    }
}

string strat2name (strategy strat) {
	switch(strat) {
		case    MINIMAX    :    return "Minimax";
		case    RANDOM     :    return "Random";
		case    QTAB       :    return "Single Q-learning (Tabular)";
		case    DOUBLEQ    :    return "Double Q-learning (Tabular)";
		case    DQN        :    return "DQN";
		case    HEURIBOT   :    return "Heuribot";
		default            :    return "ERROR: Invalid strategy";
	}
}
