#include <iostream>
#include <cctype>
#include <fstream>
#include "cooke_parser.h"
#include "cooke_analyzer.h"



using namespace std;

int nextToken;

static int charClass;
char lexeme[100];
static char nextChar;
static int lexLen;
int lineNumber = 1;
static ifstream in_fp;


static void addChar();


static void getChar();

static void getNonBlank();




const char* getName(int token);









static int lookup(char ch)
{
    switch (ch)
    {
    case '(':
        addChar();
        nextToken = LEFT_PAREN;
        break;
    case ')':
        addChar();
        nextToken = RIGHT_PAREN;
        break;
    case '+':
        addChar();
        getChar();
        if (nextChar == '+')
        {
            addChar();
            nextToken = INC_OP;
            break;
        }
        else
        {
            in_fp.unget();
            nextToken = ADD_OP;
            break;

        }
        
    case '-':
        addChar();
        getChar();
        if (nextChar == '-')
        {
            addChar();
            nextToken = DEC_OP;
            break;
        }
        else
        {
            in_fp.unget();
            nextToken = SUB_OP;
            break;
        }
        
    case '*':
        addChar();
        getChar();
        if (nextChar == '*') {
            addChar();
            nextToken = POW_OP;
        }
        else {
            in_fp.unget();
            nextToken = MULT_OP;
        }
        break;
    case '/':
        addChar();
        nextToken = DIV_OP;
        break;



    case'<':
        addChar();
        getChar();
        if (nextChar== '>')
        {
            addChar();
            nextToken = NEQUAL_OP;
            break;
        }
        else if (nextChar == '=') {
            addChar();
            nextToken = LEQUAL_OP;
            break;
        }
        else
            
        {
         
            nextToken = LESSER_OP;
            in_fp.unget();
            break;

        }

    case'>':
        addChar();
        getChar();
        if (nextChar == '=')
        {
            addChar();
            nextToken = GEQUAL_OP;
            break;
        }
        else
        {
            in_fp.unget();
            nextToken = GREATER_OP;
            break;

        }


    case'=':
        addChar();
        nextToken = EQUAL_OP;
        break;




    case':':
        addChar();
        getChar();  
        if (nextChar== '=')
        {
            addChar();
            nextToken = ASSIGN_OP;
            break;
        }
         else
        {
            in_fp.unget();
            nextToken = COLON;
            break;
        }
       
    case ';':
        addChar();
        nextToken = SEMICOLON;
        break;






    
    
    
    
    
    
    
    
    
    
    
    
    
    
   
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    default:
        addChar();
        nextToken = UNKNOWN;
        break;





    }
    return nextToken;
}




    
















static void addChar()
{
    if (lexLen <= 98)
    {
        lexeme[lexLen++] = nextChar;
        lexeme[lexLen] = 0;
    }
    else
    {   
        cout << ("Error - Lexeme is to long \n ");
    }
    
}





















static void getChar() {
    if ((nextChar = in_fp.get()) != EOF) {
        if (isalpha(nextChar))
            charClass = LETTER;
        else if (isdigit(nextChar))
            charClass = DIGIT;
        else
            charClass = UNKNOWN;
    }
    else {
        charClass = EOF;
    }
}







static void getNonBlank() {
    while (isspace(nextChar))
    {
        if (nextChar == '\n')
        {
            lineNumber++;
        }
        getChar();
   
    }
    
}













int lex()
{
    lexLen = 0;
    getNonBlank();

    switch (charClass)
    {
    case LETTER:
        addChar();
        getChar();
        while (charClass == LETTER || charClass == DIGIT)
        {
            addChar();
            getChar();
        }
        if (lexeme[0] == 'r' && lexeme[1] == 'e' && lexeme[2] == 'a' && lexeme[3] == 'd' && lexeme[4] == '\0')
        {
            nextToken = KEY_READ;
        }
        else if (lexeme[0] == 'p' && lexeme[1] == 'r' && lexeme[2] == 'i' && lexeme[3] == 'n' && lexeme[4] == 't' && lexeme[5] == '\0')
        {
            nextToken = KEY_PRINT;
        }
        else if (lexeme[0] == 'i' && lexeme[1] == 'f' && lexeme[2] == '\0') {
            nextToken = KEY_IF;
        }
        else if (lexeme[0] == 'e' && lexeme[1] == 'l' && lexeme[2] == 's' && lexeme[3] == 'e' && lexeme[4] == '\0') 
        {
            nextToken = KEY_ELSE;
        }
        else if (lexeme[0] == 'b' && lexeme[1] == 'e' && lexeme[2] == 'g' && lexeme[3] == 'i' && lexeme[4] == 'n' && lexeme[5] == '\0') 
        {
            nextToken = KEY_BEGIN;
        }
        else if (lexeme[0] == 'e' && lexeme[1] == 'n' && lexeme[2] == 'd' && lexeme[3] == '\0') 
        {
            nextToken = KEY_END;
        }
        else {
            nextToken = IDENT;
        }
        break;



    case DIGIT:
        addChar();
        getChar();
        while (charClass == DIGIT)
        {
            addChar();
            getChar();
        }
        nextToken = INT_LIT;
        break;

    case UNKNOWN:
        lookup(nextChar);
        getChar();
        break;

        


        


    case EOF:
        nextToken = EOF;
        lexeme[0] = 'E';
        lexeme[1] = 'O';
        lexeme[2] = 'F';
        lexeme[3] = 0;
        break;
    }

    if (nextToken != -1) {
        printf("%s\t%s\n", lexeme, getName(nextToken));
    }

    return nextToken;
}


const char* getName(int token) {


    switch (token) {

    case ASSIGN_OP:
        return "ASSIGN_OP";
        break;


    case ADD_OP:
        return "ADD_OP";
        break;

    case KEY_READ:
        return "KEY_READ";
        break;

    case LESSER_OP:
        return "LESSER_OP";
        break;

    case SUB_OP:
        return "SUB_OP";
        break;

    case KEY_PRINT:
        return "KEY_PRINT";
        break;

    case GREATER_OP:
        return "GREATER_OP";
        break;

    case MULT_OP:
        return "MULT_OP";
        break;

    case KEY_IF:
        return "KEY_IF";
        break;

    case EQUAL_OP:
        return "EQUAL_OP";
        break;

    case DIV_OP:
        return "DIV_OP";
        break;

    case KEY_ELSE:
        return "KEY_ELSE";
        break;

    case NEQUAL_OP:
        return "NEQUAL_OP";
        break;

    case POW_OP:
        return "POW_OP";
        break;

    case KEY_BEGIN:
        return "KEY_BEGIN";
        break;

    case LEQUAL_OP:
        return "LEQUAL_OP";
        break;

    case INC_OP:
        return "INC_OP";
        break;

    case KEY_END:
        return "KEY_END";
        break;

    case GEQUAL_OP:
        return "GEQUAL_OP";
        break;

    case DEC_OP:
        return "DEC_OP";
        break;


    case IDENT:
        return "IDENT";
        break;


    case LEFT_PAREN:
        return "LEFT_PAREN";
        break;

    case SEMICOLON:
        return "SEMICOLON";
        break;

    case INT_LIT:
        return "INT_LIT";
        break;

    case RIGHT_PAREN:
        return "RIGHT_PAREN";
        break;


    case COLON:
        return "COLON";
        break;



    default:
        return "UNKNOWN";
        break;
    }
}




























   int main(int argc, char* argv[])
    {
        if (argc != 2) {
            cout << "Error: You must provide a source code file. \n" << endl;
            return 2;
        }

        in_fp.open(argv[1]);
        if (!in_fp) {
            cout << "Error: Cannot open file ' \n" << argv[1] << "'" << endl;
            return 3;
        }

        cout << "Cooke Parser :: R11921013 \n"  << endl;

        getChar();  
        lex();      

        P();       

        cout << "Syntax Validated" << endl;

        in_fp.close();
        return 0;
   }

        




