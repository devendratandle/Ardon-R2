// R2 Parser — recursive descent, produces Expr AST
// Both <- and = work for assignment (user's choice)
// = inside function calls is named argument (context-sensitive)

use crate::lexer::{Token, Tok, Lexer, LexErr};
use r2_types::*;
use std::sync::Arc;

pub struct Parser { tokens: Vec<Tok>, pos: usize }

#[derive(Debug)]
pub struct ParseErr { pub msg: String, pub line: usize, pub col: usize }
impl std::fmt::Display for ParseErr { fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result { write!(f, "Parse error at {}:{}: {}", self.line, self.col, self.msg) } }
impl std::error::Error for ParseErr {}
impl From<LexErr> for ParseErr { fn from(e: LexErr) -> Self { ParseErr { msg: e.msg, line: e.line, col: e.col } } }

impl ParseErr {
    /// Render the error with the offending source line + `^` underline
    /// pointing at the failing column. Falls back to the plain message
    /// when `source` is empty or the line is out of range.
    ///
    /// Example output:
    /// ```text
    /// Parse error at 3:12: expected ')', got Number(3.0)
    ///   3 | mean(c(1, 2, 3
    ///                 ^
    /// ```
    pub fn display_with_source(&self, source: &str) -> String {
        let lines: Vec<&str> = source.lines().collect();
        if self.line == 0 || self.line > lines.len() {
            return format!("{}", self);
        }
        let line_text = lines[self.line - 1];
        let lineno_width = self.line.to_string().len();
        let prefix_width = lineno_width + 3; // "  N | "
        let caret_col = self.col.saturating_sub(1);
        let mut out = String::new();
        out.push_str(&format!("{}\n", self));
        out.push_str(&format!("  {:>width$} | {}\n", self.line, line_text, width = lineno_width));
        out.push_str(&format!("{}{}^\n", " ".repeat(prefix_width), " ".repeat(caret_col)));
        out
    }
}

impl Parser {
    pub fn parse(source: &str) -> Result<Vec<Expr>, ParseErr> {
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize()?;
        let mut p = Parser { tokens, pos: 0 };
        p.parse_program()
    }

    fn peek(&self) -> &Token { self.tokens.get(self.pos).map(|t| &t.token).unwrap_or(&Token::Eof) }
    fn loc(&self) -> (usize, usize) { self.tokens.get(self.pos).map(|t| (t.line, t.col)).unwrap_or((0,0)) }
    fn advance(&mut self) -> Token { let t = self.tokens[self.pos].token.clone(); self.pos += 1; t }
    fn expect(&mut self, expected: &Token) -> Result<(), ParseErr> {
        if self.peek() == expected { self.advance(); Ok(()) }
        else { let (l,c) = self.loc(); Err(ParseErr { msg: format!("expected {:?}, got {:?}", expected, self.peek()), line: l, col: c }) }
    }
    fn skip_nl(&mut self) { while matches!(self.peek(), Token::Newline | Token::Comment(_)) { self.advance(); } }

    fn parse_program(&mut self) -> Result<Vec<Expr>, ParseErr> {
        let mut stmts = Vec::new();
        self.skip_nl();
        while !matches!(self.peek(), Token::Eof) {
            stmts.push(self.parse_expr()?);
            self.skip_nl();
            if matches!(self.peek(), Token::Semi) { self.advance(); }
            self.skip_nl();
        }
        Ok(stmts)
    }

    fn parse_expr(&mut self) -> Result<Expr, ParseErr> {
        match self.peek() {
            Token::Type => return self.parse_type_def(),
            Token::Method => return self.parse_method_def(),
            _ => {}
        }
        self.parse_assign()
    }

    // ── Assignment: x <- value OR x = value ──────────────────────────
    fn parse_assign(&mut self) -> Result<Expr, ParseErr> {
        let lhs = self.parse_pipe()?;
        self.skip_nl();
        match self.peek() {
            Token::Arrow | Token::SuperArrow => {
                self.advance(); self.skip_nl();
                let rhs = self.parse_assign()?;
                Ok(Expr::Assign { target: Box::new(lhs), value: Box::new(rhs) })
            }
            Token::Equals => {
                // = is assignment at top level / statement context
                // Inside call args it's handled by parse_call_args before we get here
                self.advance(); self.skip_nl();
                let rhs = self.parse_assign()?;
                Ok(Expr::Assign { target: Box::new(lhs), value: Box::new(rhs) })
            }
            Token::RightArrow => {
                self.advance(); self.skip_nl();
                let rhs = self.parse_assign()?;
                Ok(Expr::Assign { target: Box::new(rhs), value: Box::new(lhs) })
            }
            _ => Ok(lhs),
        }
    }

    fn parse_pipe(&mut self) -> Result<Expr, ParseErr> {
        let mut lhs = self.parse_tilde()?;
        loop { self.skip_nl(); if matches!(self.peek(), Token::Pipe) { self.advance(); self.skip_nl(); let rhs = self.parse_tilde()?; lhs = Expr::Pipe { lhs: Box::new(lhs), rhs: Box::new(rhs) }; } else { break; } }
        Ok(lhs)
    }

    fn parse_tilde(&mut self) -> Result<Expr, ParseErr> {
        if matches!(self.peek(), Token::Tilde) { self.advance(); self.skip_nl(); let rhs = self.parse_or()?; return Ok(Expr::Binary { op: BinOp::Tilde, lhs: Box::new(Expr::NullLit), rhs: Box::new(rhs) }); }
        let lhs = self.parse_or()?;
        if matches!(self.peek(), Token::Tilde) { self.advance(); self.skip_nl(); let rhs = self.parse_or()?; Ok(Expr::Binary { op: BinOp::Tilde, lhs: Box::new(lhs), rhs: Box::new(rhs) }) }
        else { Ok(lhs) }
    }

    fn parse_or(&mut self) -> Result<Expr, ParseErr> {
        let mut lhs = self.parse_and()?;
        loop { let op = match self.peek() { Token::Or => BinOp::Or, Token::OrShort => BinOp::OrShort, _ => break }; self.advance(); self.skip_nl(); let rhs = self.parse_and()?; lhs = Expr::Binary { op, lhs: Box::new(lhs), rhs: Box::new(rhs) }; }
        Ok(lhs)
    }
    fn parse_and(&mut self) -> Result<Expr, ParseErr> {
        let mut lhs = self.parse_not()?;
        loop { let op = match self.peek() { Token::And => BinOp::And, Token::AndShort => BinOp::AndShort, _ => break }; self.advance(); self.skip_nl(); let rhs = self.parse_not()?; lhs = Expr::Binary { op, lhs: Box::new(lhs), rhs: Box::new(rhs) }; }
        Ok(lhs)
    }
    fn parse_not(&mut self) -> Result<Expr, ParseErr> {
        if matches!(self.peek(), Token::Bang) { self.advance(); self.skip_nl(); let e = self.parse_not()?; return Ok(Expr::Unary { op: UnOp::Not, expr: Box::new(e) }); }
        self.parse_compare()
    }
    fn parse_compare(&mut self) -> Result<Expr, ParseErr> {
        let mut lhs = self.parse_add()?;
        loop { let op = match self.peek() { Token::Eq => BinOp::Eq, Token::Ne => BinOp::Ne, Token::Lt => BinOp::Lt, Token::Gt => BinOp::Gt, Token::Le => BinOp::Le, Token::Ge => BinOp::Ge, _ => break }; self.advance(); self.skip_nl(); let rhs = self.parse_add()?; lhs = Expr::Binary { op, lhs: Box::new(lhs), rhs: Box::new(rhs) }; }
        Ok(lhs)
    }
    fn parse_add(&mut self) -> Result<Expr, ParseErr> {
        let mut lhs = self.parse_mul()?;
        loop { let op = match self.peek() { Token::Plus => BinOp::Add, Token::Minus => BinOp::Sub, _ => break }; self.advance(); self.skip_nl(); let rhs = self.parse_mul()?; lhs = Expr::Binary { op, lhs: Box::new(lhs), rhs: Box::new(rhs) }; }
        Ok(lhs)
    }
    fn parse_mul(&mut self) -> Result<Expr, ParseErr> {
        let mut lhs = self.parse_colon()?;
        loop {
            // `%name%` infix operator (e.g. `x %in% y`) desugars to a call of
            // the function named by the operator: `\`%in%\`(x, y)`.
            if let Token::Infix(name) = self.peek().clone() {
                self.advance(); self.skip_nl();
                let rhs = self.parse_colon()?;
                lhs = Expr::Call {
                    func: Box::new(Expr::Symbol(Arc::from(name.as_str()))),
                    args: vec![
                        CallArg { name: None, value: lhs },
                        CallArg { name: None, value: rhs },
                    ],
                };
                continue;
            }
            let op = match self.peek() { Token::Star => BinOp::Mul, Token::Slash => BinOp::Div, Token::Percent => BinOp::Mod, Token::IntDiv => BinOp::IntDiv, Token::MatMul => BinOp::MatMul, _ => break }; self.advance(); self.skip_nl(); let rhs = self.parse_colon()?; lhs = Expr::Binary { op, lhs: Box::new(lhs), rhs: Box::new(rhs) }; }
        Ok(lhs)
    }
    fn parse_colon(&mut self) -> Result<Expr, ParseErr> {
        let mut lhs = self.parse_unary()?;
        while matches!(self.peek(), Token::Colon) { self.advance(); self.skip_nl(); let rhs = self.parse_unary()?; lhs = Expr::Binary { op: BinOp::Colon, lhs: Box::new(lhs), rhs: Box::new(rhs) }; }
        Ok(lhs)
    }
    fn parse_unary(&mut self) -> Result<Expr, ParseErr> {
        match self.peek() {
            Token::Minus => { self.advance(); self.skip_nl(); let e = self.parse_power()?; Ok(Expr::Unary { op: UnOp::Neg, expr: Box::new(e) }) }
            Token::Plus => { self.advance(); self.skip_nl(); self.parse_power() }
            _ => self.parse_power(),
        }
    }
    fn parse_power(&mut self) -> Result<Expr, ParseErr> {
        let base = self.parse_postfix()?;
        if matches!(self.peek(), Token::Caret) { self.advance(); self.skip_nl(); let exp = self.parse_unary()?; Ok(Expr::Binary { op: BinOp::Pow, lhs: Box::new(base), rhs: Box::new(exp) }) }
        else { Ok(base) }
    }

    fn parse_postfix(&mut self) -> Result<Expr, ParseErr> {
        let mut obj = self.parse_atom()?;
        loop { match self.peek() {
            Token::LParen => { self.advance(); let args = self.parse_call_args()?; self.expect(&Token::RParen)?; obj = Expr::Call { func: Box::new(obj), args }; }
            Token::LBrack => { self.advance(); self.skip_nl(); let mut indices = Vec::new();
                while !matches!(self.peek(), Token::RBrack) {
                    if matches!(self.peek(), Token::Comma) { indices.push(None); } else { indices.push(Some(self.parse_expr()?)); }
                    if matches!(self.peek(), Token::Comma) { self.advance(); self.skip_nl();
                        // If next is ] after comma, push None for empty trailing index (e.g. iris[1:10,])
                        if matches!(self.peek(), Token::RBrack) { indices.push(None); }
                    }
                }
                self.expect(&Token::RBrack)?; obj = Expr::Index { object: Box::new(obj), indices }; }
            Token::LDblBrack => { self.advance(); self.skip_nl(); let idx = self.parse_expr()?; self.skip_nl(); self.expect(&Token::RDblBrack)?; obj = Expr::DblIndex { object: Box::new(obj), index: Box::new(idx) }; }
            Token::Dollar => { self.advance(); if let Token::Ident(name) = self.peek().clone() { self.advance(); obj = Expr::Dollar { object: Box::new(obj), field: Arc::from(name.as_str()) }; } else { let (l,c) = self.loc(); return Err(ParseErr { msg: "expected name after $".into(), line: l, col: c }); } }
            Token::DblColon => { self.advance(); if let (Expr::Symbol(pkg), Token::Ident(name)) = (&obj, self.peek().clone()) { let pkg = pkg.clone(); self.advance(); obj = Expr::Namespace { pkg, name: Arc::from(name.as_str()) }; } else { let (l,c) = self.loc(); return Err(ParseErr { msg: "expected pkg::name".into(), line: l, col: c }); } }
            _ => break,
        }}
        Ok(obj)
    }

    // Inside function calls: name = value is a NAMED ARGUMENT, not assignment
    // (see parse_call_args). `soft_keyword_name` below lets R2's soft
    // keywords double as R argument names.
    fn parse_call_args(&mut self) -> Result<Vec<CallArg>, ParseErr> {
        let mut args = Vec::new();
        self.skip_nl();
        if matches!(self.peek(), Token::RParen) { return Ok(args); }
        loop {
            self.skip_nl();
            // A named argument is `<name> = value`. The name is normally an
            // identifier, but R has no reserved words — so the R2-specific
            // soft keywords (`type`, `method`, `extends`, …) must also be
            // usable as argument names. We only treat them as a name when
            // immediately followed by `=`, so `try(x)` / `match(x)` in
            // argument position still parse as expressions. `type=` is the
            // common case (e.g. plot(x, y, type="l")).
            let name_text: Option<String> = match self.peek() {
                Token::Ident(s) => Some(s.clone()),
                other => soft_keyword_name(other).map(|s| s.to_string()),
            };
            let arg = if let Some(name) = name_text {
                let saved = self.pos;
                self.advance();
                if matches!(self.peek(), Token::Equals) {
                    // name = value → named argument (NOT assignment)
                    self.advance(); self.skip_nl();
                    CallArg { name: Some(Arc::from(name.as_str())), value: self.parse_expr()? }
                } else {
                    self.pos = saved;
                    CallArg { name: None, value: self.parse_expr()? }
                }
            } else {
                CallArg { name: None, value: self.parse_expr()? }
            };
            args.push(arg);
            self.skip_nl();
            if matches!(self.peek(), Token::Comma) { self.advance(); } else { break; }
        }
        Ok(args)
    }

    fn parse_atom(&mut self) -> Result<Expr, ParseErr> {
        let tok = self.peek().clone();
        match tok {
            Token::Number(n) => { self.advance(); Ok(Expr::NumLit(n)) }
            Token::Int(n) => { self.advance(); Ok(Expr::IntLit(n)) }
            Token::Str(s) => { self.advance(); Ok(Expr::StrLit(s)) }
            Token::FStr(s) => { self.advance(); Ok(Expr::FStringLit(parse_fstring_parts(&s)?)) }
            Token::True => { self.advance(); Ok(Expr::BoolLit(true)) }
            Token::False => { self.advance(); Ok(Expr::BoolLit(false)) }
            Token::Na => { self.advance(); Ok(Expr::NaLit) }
            Token::Null => { self.advance(); Ok(Expr::NullLit) }
            Token::Inf => { self.advance(); Ok(Expr::NumLit(f64::INFINITY)) }
            Token::NaN => { self.advance(); Ok(Expr::NumLit(f64::NAN)) }
            Token::Ident(s) => { self.advance(); Ok(Expr::Symbol(Arc::from(s.as_str()))) }
            Token::DotDotDot => { self.advance(); Ok(Expr::Dots) }
            Token::LParen => { self.advance(); self.skip_nl(); let e = self.parse_expr()?; self.skip_nl(); self.expect(&Token::RParen)?; Ok(e) }
            Token::LBrace => self.parse_block(),
            Token::If => self.parse_if(),
            Token::For => self.parse_for(),
            Token::While => self.parse_while(),
            Token::Match => self.parse_match(),
            Token::Function => self.parse_function(),
            Token::Backslash => self.parse_lambda(),
            Token::Try => self.parse_try_catch(),
            Token::Return => { self.advance(); if matches!(self.peek(), Token::LParen) { self.advance(); self.skip_nl(); let v = if matches!(self.peek(), Token::RParen) { Expr::NullLit } else { self.parse_expr()? }; self.skip_nl(); self.expect(&Token::RParen)?; Ok(Expr::Return(Box::new(v))) } else { Ok(Expr::Return(Box::new(Expr::NullLit))) } }
            Token::Break => { self.advance(); Ok(Expr::Break) }
            Token::Next => { self.advance(); Ok(Expr::Next) }
            _ => { let (l,c) = self.loc(); Err(ParseErr { msg: format!("unexpected: {:?}", tok), line: l, col: c }) }
        }
    }

    fn parse_block(&mut self) -> Result<Expr, ParseErr> {
        self.expect(&Token::LBrace)?; self.skip_nl();
        let mut stmts = Vec::new();
        while !matches!(self.peek(), Token::RBrace) { stmts.push(self.parse_expr()?); self.skip_nl(); if matches!(self.peek(), Token::Semi) { self.advance(); } self.skip_nl(); }
        self.expect(&Token::RBrace)?; Ok(Expr::Block(stmts))
    }
    fn parse_if(&mut self) -> Result<Expr, ParseErr> {
        self.advance(); self.expect(&Token::LParen)?; self.skip_nl(); let cond = self.parse_expr()?; self.skip_nl(); self.expect(&Token::RParen)?; self.skip_nl(); let then = self.parse_expr()?; self.skip_nl();
        let else_ = if matches!(self.peek(), Token::Else) { self.advance(); self.skip_nl(); Some(Box::new(self.parse_expr()?)) } else { None };
        Ok(Expr::If { cond: Box::new(cond), then: Box::new(then), else_ })
    }
    fn parse_for(&mut self) -> Result<Expr, ParseErr> {
        self.advance(); self.expect(&Token::LParen)?;
        let var = match self.advance() { Token::Ident(s) => Arc::from(s.as_str()), _ => { let (l,c) = self.loc(); return Err(ParseErr { msg: "expected variable".into(), line: l, col: c }); } };
        self.expect(&Token::In)?; self.skip_nl(); let iter = self.parse_expr()?; self.skip_nl(); self.expect(&Token::RParen)?; self.skip_nl(); let body = self.parse_expr()?;
        Ok(Expr::For { var, iter: Box::new(iter), body: Box::new(body) })
    }
    fn parse_while(&mut self) -> Result<Expr, ParseErr> {
        self.advance(); self.expect(&Token::LParen)?; self.skip_nl(); let cond = self.parse_expr()?; self.skip_nl(); self.expect(&Token::RParen)?; self.skip_nl(); let body = self.parse_expr()?;
        Ok(Expr::While { cond: Box::new(cond), body: Box::new(body) })
    }
    fn parse_match(&mut self) -> Result<Expr, ParseErr> {
        self.advance(); self.expect(&Token::LParen)?; self.skip_nl(); let expr = self.parse_expr()?; self.skip_nl(); self.expect(&Token::RParen)?; self.skip_nl(); self.expect(&Token::LBrace)?; self.skip_nl();
        let mut arms = Vec::new();
        while !matches!(self.peek(), Token::RBrace) {
            // Parse patterns using parse_pipe (NOT parse_expr) so -> isn't consumed as right-assignment
            let mut patterns = vec![self.parse_pipe()?];
            while matches!(self.peek(), Token::Comma) { self.advance(); self.skip_nl(); if matches!(self.peek(), Token::RBrace) { break; } patterns.push(self.parse_pipe()?); }
            self.skip_nl(); self.expect(&Token::RightArrow)?; self.skip_nl();
            // Body IS a full expression (assignment allowed in body)
            let body = self.parse_pipe()?; arms.push(MatchArm { patterns, body });
            self.skip_nl(); if matches!(self.peek(), Token::Comma) { self.advance(); } self.skip_nl();
        }
        self.expect(&Token::RBrace)?; Ok(Expr::Match { expr: Box::new(expr), arms })
    }
    fn parse_function(&mut self) -> Result<Expr, ParseErr> {
        self.advance(); self.expect(&Token::LParen)?; let params = self.parse_params()?; self.expect(&Token::RParen)?; self.skip_nl(); let body = self.parse_expr()?;
        Ok(Expr::FuncDef { params, body: Box::new(body) })
    }
    fn parse_lambda(&mut self) -> Result<Expr, ParseErr> {
        self.advance(); self.expect(&Token::LParen)?; let params = self.parse_params()?; self.expect(&Token::RParen)?; self.skip_nl(); let body = self.parse_expr()?;
        Ok(Expr::Lambda { params, body: Box::new(body) })
    }
    fn parse_params(&mut self) -> Result<Vec<Param>, ParseErr> {
        let mut params = Vec::new(); self.skip_nl();
        while !matches!(self.peek(), Token::RParen) {
            if matches!(self.peek(), Token::DotDotDot) { self.advance(); params.push(Param { name: Arc::from("..."), default: None, dots: true }); }
            else if let Token::Ident(name) = self.peek().clone() {
                self.advance();
                let default = if matches!(self.peek(), Token::Equals) { self.advance(); self.skip_nl(); Some(Box::new(self.parse_expr()?)) } else { None };
                params.push(Param { name: Arc::from(name.as_str()), default, dots: false });
            }
            self.skip_nl(); if matches!(self.peek(), Token::Comma) { self.advance(); self.skip_nl(); }
        }
        Ok(params)
    }
    fn parse_try_catch(&mut self) -> Result<Expr, ParseErr> {
        self.advance(); self.skip_nl(); let body = self.parse_block()?; self.skip_nl();
        self.expect(&Token::Catch)?; self.expect(&Token::LParen)?;
        let var = match self.advance() { Token::Ident(s) => Arc::from(s.as_str()), _ => { let (l,c) = self.loc(); return Err(ParseErr { msg: "expected variable in catch".into(), line: l, col: c }); } };
        self.expect(&Token::RParen)?; self.skip_nl(); let catch = self.parse_block()?;
        Ok(Expr::TryCatch { body: Box::new(body), var, catch: Box::new(catch) })
    }
    fn parse_type_def(&mut self) -> Result<Expr, ParseErr> {
        self.advance();
        let name = match self.advance() { Token::Ident(s) => Arc::from(s.as_str()), _ => { let (l,c) = self.loc(); return Err(ParseErr { msg: "expected type name".into(), line: l, col: c }); } };
        let parent = if matches!(self.peek(), Token::Extends) { self.advance(); match self.advance() { Token::Ident(s) => Some(Arc::from(s.as_str())), _ => { let (l,c) = self.loc(); return Err(ParseErr { msg: "expected parent name".into(), line: l, col: c }); } } } else { None };
        self.skip_nl(); self.expect(&Token::LBrace)?; self.skip_nl();
        let mut fields = Vec::new();
        while !matches!(self.peek(), Token::RBrace) {
            let fname = match self.advance() { Token::Ident(s) => Arc::from(s.as_str()), _ => break };
            self.expect(&Token::Colon)?;
            let ftype = match self.advance() { Token::Ident(s) => match s.as_str() { "numeric" => FieldType::Numeric, "integer" => FieldType::Integer, "character" => FieldType::Character, "logical" => FieldType::Logical, "tensor" => FieldType::Tensor, "matrix" => FieldType::Matrix, "any" => FieldType::Any, other => FieldType::TypeRef(Arc::from(other)) }, _ => FieldType::Any };
            fields.push(FieldDef { name: fname, field_type: ftype, default: None });
            self.skip_nl(); if matches!(self.peek(), Token::Comma) { self.advance(); } self.skip_nl();
        }
        self.expect(&Token::RBrace)?; Ok(Expr::TypeDef { name, fields, parent })
    }
    fn parse_method_def(&mut self) -> Result<Expr, ParseErr> {
        self.advance();
        let method_name = match self.advance() { Token::Ident(s) => Arc::from(s.as_str()), _ => { let (l,c) = self.loc(); return Err(ParseErr { msg: "expected method name".into(), line: l, col: c }); } };
        self.expect(&Token::LParen)?;
        let param_name = match self.advance() { Token::Ident(s) => Arc::from(s.as_str()), _ => { let (l,c) = self.loc(); return Err(ParseErr { msg: "expected param".into(), line: l, col: c }); } };
        self.expect(&Token::Colon)?;
        let type_name = match self.advance() { Token::Ident(s) => Arc::from(s.as_str()), _ => { let (l,c) = self.loc(); return Err(ParseErr { msg: "expected type".into(), line: l, col: c }); } };
        let mut extra = Vec::new();
        if matches!(self.peek(), Token::Comma) { self.advance(); self.skip_nl();
            while !matches!(self.peek(), Token::RParen) {
                if let Token::Ident(n) = self.peek().clone() { self.advance(); let d = if matches!(self.peek(), Token::Equals) { self.advance(); self.skip_nl(); Some(Box::new(self.parse_expr()?)) } else { None }; extra.push(Param { name: Arc::from(n.as_str()), default: d, dots: false }); }
                self.skip_nl(); if matches!(self.peek(), Token::Comma) { self.advance(); self.skip_nl(); }
            }
        }
        self.expect(&Token::RParen)?; self.skip_nl(); let body = self.parse_expr()?;
        Ok(Expr::MethodDef(Method { name: method_name, type_name, param_name, extra_params: extra, body: Box::new(body) }))
    }
}

/// The R2-specific soft keywords that may also appear as R argument names
/// (R has no reserved words). Returns the keyword's spelling so the call
/// parser can use it as a named-argument name when followed by `=`.
/// `match`/`try`/`catch` are intentionally excluded — in argument position
/// they introduce real expressions, and are not used as R argument names.
fn soft_keyword_name(tok: &Token) -> Option<&'static str> {
    Some(match tok {
        Token::Type    => "type",
        Token::Method  => "method",
        Token::Extends  => "extends",
        Token::Strict  => "strict",
        Token::Lenient => "lenient",
        _ => return None,
    })
}

fn parse_fstring_parts(s: &str) -> Result<Vec<FStringPart>, ParseErr> {
    let mut parts = Vec::new(); let mut lit = String::new(); let chars: Vec<char> = s.chars().collect(); let mut i = 0;
    while i < chars.len() {
        if chars[i] == '{' {
            if i + 1 < chars.len() && chars[i+1] == '{' { lit.push('{'); i += 2; continue; }
            if !lit.is_empty() { parts.push(FStringPart::Literal(std::mem::take(&mut lit))); }
            i += 1; let mut expr_s = String::new(); let mut depth = 1;
            while i < chars.len() && depth > 0 { if chars[i] == '{' { depth += 1; } if chars[i] == '}' { depth -= 1; if depth == 0 { break; } } expr_s.push(chars[i]); i += 1; }
            i += 1;
            if let Some(expr) = Parser::parse(&expr_s)?.into_iter().next() { parts.push(FStringPart::Expr(expr)); }
        } else if chars[i] == '}' && i + 1 < chars.len() && chars[i+1] == '}' { lit.push('}'); i += 2; }
        else { lit.push(chars[i]); i += 1; }
    }
    if !lit.is_empty() { parts.push(FStringPart::Literal(lit)); }
    Ok(parts)
}
