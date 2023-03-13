#include <iostream>
using namespace std;
#include "Graph.h"


Graph::Graph()
{
	head_ = nullptr;
	size_ = 0;
	top_ = -1;
	front_ = -1;
	rear_ = -1;
}


int Graph::add_node(const char value)
{
	node* pos_node;

	this->temp_ = new node;
	this->temp_->nom = value;
	this->temp_->next_node = nullptr;
	this->temp_->next = nullptr;

	if (!this->head_)
	{
		this->head_ = temp_;
		this->size_++;
		return 0;
	}

	const int exists = search_node(value, pos_node);

	if (exists == 0)
	{
		pos_node->next_node = temp_;
		this->size_++;
		return 0;
	}
	delete temp_;
	return -1;
}


int Graph::add_edge(const char org, const char dest, const int weight)
{
	node *org_pos = nullptr, *dest_pos = nullptr;
	edge* ady_pos = nullptr;

	this->ady_ = new edge;
	this->ady_->nom = dest;
	this->ady_->weight = weight;
	this->ady_->next = nullptr;

	if (!this->head_)
		return -1;

	const int org_exists = search_node(org, org_pos);
	const int dest_exists = search_node(dest, dest_pos);

	if (org_exists == 0 || dest_exists == 0)
	{
		delete ady_;
		return -1;
	}

	if (!org_pos->next)
	{
		org_pos->next = ady_;
		ady_->next_node = dest_pos;
		return 0;
	}

	const int ady_exists = search_edge(org_pos, dest, ady_pos);


	if (ady_exists == -1)
	{
		delete ady_;
		return -2;
	}
	ady_pos->next = ady_;
	ady_->next_node = dest_pos;
	return 0;
}


int Graph::search_node(char name, node*& pos)
{
	this->aux_ = this->head_;
	do
	{
		pos = this->aux_;
		if (this->aux_->nom == name)
			return -1;
		this->aux_ = this->aux_->next_node;
	}
	while (this->aux_);

	return 0;
}


int Graph::search_edge(node* org, char des, edge*& pos)
{
	this->aux_ady_ = org->next;

	do
	{
		pos = this->aux_ady_;
		if (this->aux_ady_->nom == des)
			return -1;
		this->aux_ady_ = this->aux_ady_->next;
	}
	while (this->aux_ady_);

	return 0;
}


void Graph::travel_breadth()
{
	queue_ = new node*[size_];
	visited_ = new node*[size_];
	int visited_count = -1;

	if (!this->head_)
	{
		cout << "Empty graph" << endl;
		return;
	}
	insert_queue(this->head_);
	node* extracted = extract_queue();

	do
	{
		cout << "\t" << extracted->nom << endl;
		visited_count++;
		visited_[visited_count] = extracted;
		edge* arc = extracted->next;
		while (arc)
		{
			int already_visited = 0;
			for (int i = 0; i <= visited_count; i++)
			{
				if (arc->next_node == visited_[i])
					already_visited = 1;
			}
			int in_queue = 0;
			for (int i = 0; i < top_; i++)
			{
				if (arc->next_node == stack_[i])
					in_queue = 1;
			}
			if (already_visited == 0 && in_queue == 0)
				insert_queue(arc->next_node);
			arc = arc->next;
		}
		extracted = extract_queue();
	}
	while (extracted);
}

void Graph::insert_queue(node* value)
{
	if (front_ == -1)
		front_++;
	rear_++;
	queue_[rear_] = value;
}

node* Graph::extract_queue()
{
	if (front_ == -1)
		return nullptr;
	if (front_ > rear_)
		return nullptr;
	front_++;
	return queue_[front_ - 1];
}
