
#include <iostream>
#include <cctype>
#include <fstream>


#include "cooke_analyzer.h"
#include "cooke_parser.h"

using namespace std;

extern const char* getName(int token);
void error(const char* msg);




void P()
{
	S();
	if (nextToken != EOF) {
		error("Unexpected tokens");
	}

};

void match(int expetedToken);


extern int nextToken;


void S()
{
	if (nextToken == EOF)
	{
		printf("Exit <S>: EOF\n");
		return;
	}

	switch (nextToken)
	{
	case IDENT:
		V();
		match(ASSIGN_OP);
		E();
		break;



	case INC_OP:
	case DEC_OP:
		O();
		break;



	case KEY_READ:                //read(V)
		match(KEY_READ);
		match(LEFT_PAREN);

		V();
		match(RIGHT_PAREN);
		break;



	case KEY_PRINT:                  //  print(E)
		match(KEY_PRINT);
		match(LEFT_PAREN);
		E();
		match(RIGHT_PAREN);
		break;



	case KEY_IF:                   // if C: begin S end | if C: begin S else: S end 
		match(KEY_IF);
		C();
		match(COLON);
		match(KEY_BEGIN);
		S();

		if (nextToken == KEY_ELSE) // this is for the ( if C: begin S else: S end) rule
		{
			match(KEY_ELSE);
			match(COLON);
			S();
		}
		match(KEY_END);
		break;


	default:
		error("Invalid statement in <S>");
	}
	if (nextToken == SEMICOLON) {
		match(SEMICOLON);
		S();
	}

	printf("Exit <S>\n");
}

void C()
{
	E();
	switch (nextToken)
	{
		case LESSER_OP:
		case GREATER_OP:
		case EQUAL_OP:
		case NEQUAL_OP:
		case LEQUAL_OP:
		case GEQUAL_OP:
			match(nextToken);
			E();
			break;
		default:
			error("Invalid  in <C>");
	}
}


void E() {
	

	T(); 

	while (nextToken == ADD_OP || nextToken == SUB_OP) {
		match(nextToken); 
		T();             
	}


}


void T() {


	F(); 

	while (nextToken == MULT_OP || nextToken == DIV_OP || nextToken == POW_OP) {
		match(nextToken);
		F();              
	}

	cout << "Exit <T>" << endl;
}



void F() {
	

	switch (nextToken) {
	case LEFT_PAREN:
		match(LEFT_PAREN);
		E();
		match(RIGHT_PAREN);
		break;

	case INT_LIT:
		N();
		break;

	case IDENT:
		V();
		break;

	default:
		error("Invalid  in <F>");
	}

	cout << "Exit <F>" << endl;
}


void O() {
	

	switch (nextToken) {
	case INC_OP:
		match(INC_OP);
		V();
		break;

	case DEC_OP:
		match(DEC_OP);
		V();
		break;

	default:
		error("Invalid  in <O>");
	}

	printf("Exit <O>\n");
}

void V() {
	if (nextToken == IDENT) {
		lex(); 
	}
	else {
		error("Invalid  in <V>");
	}
}

void N() {
	if (nextToken == INT_LIT) {
		lex(); 
	}
	else {
		error("Invalid  in <N>");
	}
}




			
	

void match(int expectedToken) {
	if (nextToken == expectedToken) {
		
		lex(); 
	}
	else {
	
		error("Unexpected token"); 
	}
}








void error(const char* msg) {
	cout << "Error encounter on line " << lineNumber
		<< ": The next lexeme was " << lexeme
		<< " and the next token was " << getName(nextToken)
		<< endl;
	exit(1);
}