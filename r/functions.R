# Crop and resize image

#' @importFrom Crop and resize an image
#'
#' @description Uses the "imager" package to crop an image to squared from and resize it
#' @param im An image as loaded by load.image
#' @param size Height and width of images in pixels
#'
#' @return A size*size*3 array
preproc.image <- function(im, size = 224) {
  # crop the image
  shape <- dim(im)
  if (shape[1] != shape[2]) {
    short.edge <- min(shape[1:2])
    xx <- floor((shape[1] - short.edge) / 2)
    yy <- floor((shape[2] - short.edge) / 2)
    im <- crop.borders(im, xx, yy)
  } 
  
  # resize to 224 x 224, needed by input of the model.
  if (dim(im)[1] != size) {
    im <- resize(im, size, size)
  }
  
  # convert to array (x, y, channel), also works with grey tone images
  array(im, dim = c(size, size, 3))
}

# Prepares image data set

#' @importFrom imager load.image 
#'
#' @description Uses "preproc.image" to prepare all images in a folder. Furthermore extracts label (0 or 1) from file name
#' @param path Folder containing n pictures
#' @param size Target height and width of images in pixels
#' @param scale Rescale picture values by that value
#' @param center Average pixel intensities on the same scale as the scaled images
#'
#' @return A list with binary response vector and size*size*3*n array
preproc.images <- function(path, size = 224, scale = 255, center = NULL) {
  stopifnot(length(fl <- list.files(path)) > 0L, size >= 2L)
  
  # Initialize array
  X <- array(NA, dim = c(size, size, 3, length(fl)))
  
  # Extract response
  y <- as.numeric(substring(fl, 1, 1))
  
  # Loop through images
  for (i in seq_along(fl)) {
    cat(".")
    im <- load.image(file.path(path, fl[i]))
    im <- try(preproc.image(im, size = size))
    if (inherits(im, "try-error")) {
      X[, , , i] <- NA
      y[i] <- NA
    } else {
      # scale and center
      if (!is.null(scale)) {
        im <- im * scale
      }
      
      if (!is.null(center)) {
        im <- im - center # need to do it within loop due to memory issues with sweep
      }
      X[, , , i] <- im
    }
  }

  list(X = X, y = y)
}
