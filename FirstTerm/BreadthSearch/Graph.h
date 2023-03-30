#pragma once
struct node;

struct edge
{
	char nom;
	int weight;
	node* next_node;
	edge* next;
};

struct node
{
	char nom;
	node* next_node;
	edge* next;
};

class Graph
{
public:
	Graph();
	int add_node(char);
	int add_edge(char, char, int);
	int search_node(char, node*&);
	int search_edge(node*, char, edge*&);
	void travel_breadth();
	void insert_queue(node*);
	node* extract_queue();

private:
	node *temp_{}, *aux_{}, *head_;
	edge *ady_{}, *aux_ady_{};
	int size_{}, top_{}, front_{}, rear_{};
	node **stack_{}, **queue_{}, **visited_{};
};
