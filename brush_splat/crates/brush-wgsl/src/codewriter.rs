pub(crate) struct CodeWriter {
    state: String,
    indent: usize,
}

impl CodeWriter {
    pub(crate) fn new() -> Self {
        Self {
            state: String::new(),
            indent: 0,
        }
    }

    pub(crate) fn add_line<T: AsRef<str>>(&mut self, line: T) {
        self.indent -= line.as_ref().chars().filter(|&c| c == '}').count();
        for _ in 0..4 * self.indent {
            self.state.push(' ');
        }
        self.state += line.as_ref();
        self.state += "\n";
        self.indent += line.as_ref().chars().filter(|&c| c == '{').count();
    }

    pub(crate) fn add_lines<T: AsRef<str>>(&mut self, lines: &[T]) {
        for line in lines {
            self.add_line(line);
        }
    }

    pub(crate) fn string(self) -> String {
        self.state
    }
}
