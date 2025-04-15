#ifndef COOKE_PARSER_H
#define COOKE_PARSER_H


#define LETTER 0
#define DIGIT 1
#define UNKNOWN 99

#define  ASSIGN_OP 11
#define  LESSER_OP 12
#define  GREATER_OP 13
#define  EQUAL_OP 14
#define  NEQUAL_OP 15	
#define  LEQUAL_OP 16
#define  GEQUAL_OP 17 
#define  LEFT_PAREN 18 
#define RIGHT_PAREN 19

#define ADD_OP 20
#define SUB_OP 21 
#define MULT_OP 22
#define DIV_OP 23
#define POW_OP 24
#define INC_OP 25
#define DEC_OP 26
#define SEMICOLON 27
#define COLON 28

#define KEY_READ 29	
#define KEY_PRINT 30
#define KEY_IF 31 
#define KEY_ELSE 32
#define KEY_BEGIN 33
#define KEY_END 34 
#define IDENT 35
#define INT_LIT 36


void P(); // Program
void S(); // Statement
void C(); // Condition
void E(); // Expression
void T(); // Term
void F(); // Factor
void V(); // Variable
void O();

void N(); // number 

void error(const char* msg);

void error(const char* msg);
extern int nextToken;
extern int lineNumber;
extern char lexeme[];

#endif /* PARSER_H */