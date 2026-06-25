// R2 Lexer — both <- and = allowed for assignment (user's choice)

use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    Number(f64), Int(i32), Str(String), FStr(String),
    True, False, Na, Null, Inf, NaN,
    Ident(String),

    // Operators
    Plus, Minus, Star, Slash, Caret, Percent, IntDiv, MatMul,
    Tilde, Bang,
    And, Or, AndShort, OrShort,
    Eq, Ne, Lt, Gt, Le, Ge,    // == != < > <= >=
    Arrow,                       // <-
    RightArrow,                  // ->
    SuperArrow,                  // <<-
    Equals,                      // = (assignment OR named arg — context decides)
    Dollar, At, Colon, DblColon,
    Pipe,                        // |>
    Backslash,                   // \ (lambda)
    DotDotDot,

    // Delimiters
    LParen, RParen, LBrack, RBrack, LDblBrack, RDblBrack,
    LBrace, RBrace, Comma, Semi, Newline,

    // Keywords
    If, Else, For, In, While, Repeat,
    Function, Return, Break, Next,
    Library, Require, Detach,
    Type, Method, Extends, Match, Try, Catch, Strict, Lenient,

    /// User/`%name%` infix operator, e.g. `%in%`, `%o%`. Holds the full
    /// spelling including the surrounding `%`.
    Infix(String),

    Comment(String), Eof,
}

#[derive(Debug, Clone)]
pub struct Tok { pub token: Token, pub line: usize, pub col: usize }

pub struct Lexer { src: Vec<char>, pos: usize, line: usize, col: usize }

#[derive(Debug, Clone)]
pub struct LexErr { pub msg: String, pub line: usize, pub col: usize }
impl fmt::Display for LexErr { fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "Syntax error at {}:{}: {}", self.line, self.col, self.msg) } }
impl std::error::Error for LexErr {}

impl Lexer {
    pub fn new(source: &str) -> Self { Lexer { src: source.chars().collect(), pos: 0, line: 1, col: 1 } }
    fn peek(&self) -> Option<char> { self.src.get(self.pos).copied() }
    fn peek2(&self) -> Option<char> { self.src.get(self.pos + 1).copied() }
    fn advance(&mut self) -> Option<char> {
        let ch = self.src.get(self.pos).copied()?;
        self.pos += 1;
        if ch == '\n' { self.line += 1; self.col = 1; } else { self.col += 1; }
        Some(ch)
    }
    fn skip_spaces(&mut self) { while matches!(self.peek(), Some(' ' | '\t' | '\r')) { self.advance(); } }

    pub fn tokenize(&mut self) -> Result<Vec<Tok>, LexErr> {
        let mut tokens = Vec::new();
        loop {
            self.skip_spaces();
            let Some(ch) = self.peek() else { tokens.push(Tok { token: Token::Eof, line: self.line, col: self.col }); break; };
            let (line, col) = (self.line, self.col);
            let tok = match ch {
                '#' => { let mut s = String::new(); self.advance(); while let Some(c) = self.peek() { if c == '\n' { break; } s.push(c); self.advance(); } Token::Comment(s) }
                '\n' => { self.advance(); Token::Newline }
                '(' => { self.advance(); Token::LParen } ')' => { self.advance(); Token::RParen }
                '{' => { self.advance(); Token::LBrace } '}' => { self.advance(); Token::RBrace }
                ',' => { self.advance(); Token::Comma } ';' => { self.advance(); Token::Semi }
                '~' => { self.advance(); Token::Tilde } '@' => { self.advance(); Token::At }
                '^' => { self.advance(); Token::Caret } '+' => { self.advance(); Token::Plus }
                '*' => { self.advance(); Token::Star } '/' => { self.advance(); Token::Slash }
                '$' => { self.advance(); Token::Dollar } '\\' => { self.advance(); Token::Backslash }
                '!' => { self.advance(); if self.peek() == Some('=') { self.advance(); Token::Ne } else { Token::Bang } }
                '<' => { self.advance(); match self.peek() {
                    Some('-') => { self.advance(); Token::Arrow }
                    Some('<') => { self.advance(); if self.peek() == Some('-') { self.advance(); Token::SuperArrow } else { return Err(LexErr { msg: "expected - after <<".into(), line, col }); } }
                    Some('=') => { self.advance(); Token::Le } _ => Token::Lt,
                }}
                '>' => { self.advance(); if self.peek() == Some('=') { self.advance(); Token::Ge } else { Token::Gt } }
                '=' => { self.advance(); if self.peek() == Some('=') { self.advance(); Token::Eq } else { Token::Equals } }
                '-' => { self.advance(); if self.peek() == Some('>') { self.advance(); Token::RightArrow } else { Token::Minus } }
                '&' => { self.advance(); if self.peek() == Some('&') { self.advance(); Token::AndShort } else { Token::And } }
                '|' => { self.advance(); match self.peek() { Some('|') => { self.advance(); Token::OrShort } Some('>') => { self.advance(); Token::Pipe } _ => Token::Or } }
                ':' => { self.advance(); if self.peek() == Some(':') { self.advance(); Token::DblColon } else { Token::Colon } }
                '%' => { self.advance(); if self.peek() == Some('/') && self.peek2() == Some('%') { self.advance(); self.advance(); Token::IntDiv } else if self.peek() == Some('*') && self.peek2() == Some('%') { self.advance(); self.advance(); Token::MatMul } else if self.peek() == Some('%') { self.advance(); Token::Percent } else if matches!(self.peek(), Some(c) if c.is_alphabetic() || c == '.') { let mut name = String::from("%"); while let Some(c) = self.peek() { if c == '%' { break; } name.push(c); self.advance(); } name.push('%'); if self.peek() == Some('%') { self.advance(); } Token::Infix(name) } else { Token::Percent } }
                '[' => { self.advance(); if self.peek() == Some('[') { self.advance(); Token::LDblBrack } else { Token::LBrack } }
                ']' => { self.advance(); if self.peek() == Some(']') { self.advance(); Token::RDblBrack } else { Token::RBrack } }
                '"' | '\'' => self.read_string(ch)?,
                'f' if self.peek2() == Some('"') => self.read_fstring()?,
                '.' if self.peek2() == Some('.') => { self.advance(); self.advance(); if self.peek() == Some('.') { self.advance(); Token::DotDotDot } else { Token::Ident("..".into()) } }
                c if c.is_ascii_digit() || (c == '.' && self.peek2().map_or(false, |d| d.is_ascii_digit())) => self.read_number()?,
                c if c.is_ascii_alphabetic() || c == '.' || c == '_' => self.read_ident(),
                '`' => self.read_backtick()?,
                _ => return Err(LexErr { msg: format!("unexpected '{}'", ch), line, col }),
            };
            tokens.push(Tok { token: tok, line, col });
        }
        Ok(tokens)
    }

    fn read_string(&mut self, q: char) -> Result<Token, LexErr> {
        self.advance(); let mut s = String::new();
        loop { match self.advance() {
            None => return Err(LexErr { msg: "unterminated string".into(), line: self.line, col: self.col }),
            Some(c) if c == q => break,
            Some('\\') => match self.advance() { Some('n') => s.push('\n'), Some('t') => s.push('\t'), Some('\\') => s.push('\\'), Some(c) => { s.push('\\'); s.push(c); } None => return Err(LexErr { msg: "unterminated escape".into(), line: self.line, col: self.col }) },
            Some(c) => s.push(c),
        }} Ok(Token::Str(s))
    }
    fn read_fstring(&mut self) -> Result<Token, LexErr> {
        self.advance(); self.advance(); let mut s = String::new();
        loop { match self.advance() {
            None => return Err(LexErr { msg: "unterminated f-string".into(), line: self.line, col: self.col }),
            Some('"') => break,
            Some('\\') => match self.advance() {
                Some('n') => s.push('\n'),
                Some('t') => s.push('\t'),
                Some('r') => s.push('\r'),
                Some('\\') => s.push('\\'),
                Some('"') => s.push('"'),
                Some('{') => s.push('{'),
                Some('}') => s.push('}'),
                Some(c) => { s.push('\\'); s.push(c); }
                None => return Err(LexErr { msg: "unterminated escape".into(), line: self.line, col: self.col }),
            },
            Some(c) => s.push(c),
        }} Ok(Token::FStr(s))
    }
    fn read_number(&mut self) -> Result<Token, LexErr> {
        let mut s = String::new(); let mut has_dot = false;
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() { s.push(c); self.advance(); }
            else if c == '.' && !has_dot { has_dot = true; s.push(c); self.advance(); }
            else if c == 'e' || c == 'E' { s.push(c); self.advance(); if matches!(self.peek(), Some('+' | '-')) { s.push(self.advance().unwrap()); } }
            else if c == 'L' && !has_dot { self.advance(); return s.parse::<i32>().map(Token::Int).map_err(|_| LexErr { msg: format!("bad int: {}", s), line: self.line, col: self.col }); }
            else { break; }
        }
        s.parse::<f64>().map(Token::Number).map_err(|_| LexErr { msg: format!("bad number: {}", s), line: self.line, col: self.col })
    }
    fn read_ident(&mut self) -> Token {
        let mut s = String::new();
        while let Some(c) = self.peek() { if c.is_ascii_alphanumeric() || c == '.' || c == '_' { s.push(c); self.advance(); } else { break; } }
        match s.as_str() {
            // Only TRUE/FALSE are reserved literals; T/F are ordinary
            // identifiers bound to TRUE/FALSE by default but reassignable (R).
            "TRUE" => Token::True, "FALSE" => Token::False,
            "NULL" => Token::Null, "NA" => Token::Na, "Inf" => Token::Inf, "NaN" => Token::NaN,
            "if" => Token::If, "else" => Token::Else, "for" => Token::For, "in" => Token::In,
            "while" => Token::While, "repeat" => Token::Repeat, "function" => Token::Function,
            "return" => Token::Return, "break" => Token::Break, "next" => Token::Next,
            "type" => Token::Type, "method" => Token::Method, "extends" => Token::Extends,
            "match" => Token::Match, "try" => Token::Try, "catch" => Token::Catch,
            // library, require, detach, strict, lenient are regular builtins, not keywords
            _ => Token::Ident(s),
        }
    }
    fn read_backtick(&mut self) -> Result<Token, LexErr> {
        self.advance(); let mut s = String::new();
        loop { match self.advance() { None => return Err(LexErr { msg: "unterminated backtick".into(), line: self.line, col: self.col }), Some('`') => break, Some(c) => s.push(c), } }
        Ok(Token::Ident(s))
    }
}
