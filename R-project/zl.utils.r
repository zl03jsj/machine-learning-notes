show_split_line <- function(title, c, count) {
    split_string = ""
    for(i in 1:count) {
    split_string <- paste(split_string, c)
    }

    split_string <- paste(split_string, title, split_string, sep="")
    print(split_string)
}

show_default_split <- function(title) {
    show_split_line(title, "-", 8)
}
