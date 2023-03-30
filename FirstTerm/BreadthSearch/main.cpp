#include <iostream>

#include "Graph.h"
using namespace std;


int main()
{
	int choice, weight, code;
	char node, org, dest;

	ady_list graph;

	do
	{
		cout << endl << endl << "MAIN MENU" << endl;
		cout << "1) Add Node" << endl;
		cout << "2) Add Edge" << endl;
		cout << "3) Show" << endl;
		cout << "4) Travel Breadth" << endl;
		cout << "0) EXIT" << endl;

		cout << "Choice: ";
		cin >> choice;

		switch (choice)
		{
		case 1:
			cout << "\tNode to insert (1 char): ";
			cin >> node;
			code = graph.add_node(node);
			if (code == 0) cout << "\tNODE INSERTED" << endl;
			if (code == -1) cout << "\tERROR... ALREADY EXISTS, NOT INSERTED" << endl;
			break;
		case 2:
			cout << "\tOrigin node (1 char): ";
			cin >> org;
			cout << "\tDestiny node (1 char): ";
			cin >> dest;
			cout << "\tEdge weight: ";
			cin >> weight;
			code = graph.add_edge(org, dest, weight);
			if (code == 0) cout << "\tEDGE INSERTED" << endl;
			if (code == -1) cout << "\tERROR... NODE DOESN'T EXIST" << endl;
			if (code == -2) cout << "\tERROR... EDGE ALREADY EXISTS, NOT INSERTED" << endl;

			code = graph.add_edge(dest, org, weight);
			if (code == 0) cout << "\tEDGE INSERTED" << endl;
			if (code == -1) cout << "\tERROR... NODE DOESN'T EXIST" << endl;
			if (code == -2) cout << "\tERROR... EDGE ALREADY EXISTS, NOT INSERTED" << endl;
			break;
		case 3:
			graph.show_full();
			break;
		case 4:
			graph.travel_breadth();
			break;
		case 0:
			cout << "EXITING..." << endl;
			break;
		default:
			cout << "INVALID OPTION..." << endl;
			break;
		}
	}
	while (choice != 0);

	return 0;
}
