#ifndef IO_H
#define IO_H

#include <stdint.h>

#include "cci.h"

/* C = client - MPI rank
 * S = server - IO daemon
 * C->S = client to server message
 * C<-S = server to client message
 */

typedef enum io_msg_type {
	CONNECT = 0,	/* C->S: rank */
	WRITE_REQ,	/* C->S: write request - length of write buffer */
	WRITE_DONE,	/* C<-S: RMA Read is done - client can reuse buffer */
	BYE,		/* C->S: client done */
	FINISHED	/* C<-S: finished writing */
} io_msg_type_t;

typedef union io_msg {
	io_msg_type_t type;	/* always first in each msg */

	struct io_msg_connect {
		io_msg_type_t type;		/* CONNECT */
		uint32_t rank;			/* client's rank */
		uint32_t ranks;			/* number of local ranks */
		uint32_t len;			/* length of buffer */
		cci_rma_handle_t handle;	/* CCI RMA handle */
	} connect;

	struct io_msg_write_request {
		io_msg_type_t type;	/* WRITE_REQ */
		uint32_t len;		/* Length of write */
		uint64_t cookie;	/* IO request opaque pointer */
	} request;

	struct io_msg_write_done {
		io_msg_type_t type;	/* WRITE_DONE */
		uint32_t pad;
		uint64_t cookie;	/* IO request opaque pointer */
	} done;

	struct io_msg_write_bye {
		io_msg_type_t type;	/* BYE */
	} bye;

	struct io_msg_write_fini {
		io_msg_type_t type;	/* FINISHED */
	} fini;
} io_msg_t;

/*!
  Init the io library

  The caller passes in an already allocated buffer and length of the
  buffer. Ideally, the buffer will be page-aligned.

  This call will:

  1) Try to start the daemon
  2) Init CCI
  3) Create an endpoint
  4) Register the buffer
  5) Marshal a connect msg
  6) Connect to the daemon

  \param[in] buffer	Data buffer to be written
  \param[in] len	Length of the buffer
  \param[in] rank	Caller's MPI rank
  \param[in] ranks	Total MPI ranks on node

  \return 0		Success, data written to daemon
  \return errno

  Each process must call once. The call must be after MPI_Init().
*/
int io_init(void *buffer, uint32_t len, uint32_t rank, uint32_t ranks);

/*!
  Write the buffer

  The semantics are similar to write(). The call is blocking until the write
  completes. Unlike write(), io_write() either succeeds completely or fails
  completely (i.e. no partial writes). As with write(), the data is buffered
  not yet on disk.

  This call will marshal and send a WRITE_REQ message to the daemon. When the
  daemon has a RMA buffer available, it will RMA Read this buffer. When the
  RMA Read completes, it will send a WRITE_DONE message back to the caller.
  The caller will poll in cci_get_event() until then.

  \param[in] len	Length to write

  \return 0		Success, data written to daemon
  \return errno

  When io_write() returns, the data has been sent to the daemon and the buffer
  may be reused.

  This call always assumes we are writing from offset 0 within the buffer passed
  to io_init().
*/
int io_write(uint32_t len);

/*!
  Release all resources

  This call will:

  1) Deregister the buffer
  2) Marshal and send a BYE msg
  3) Disconnect
  4) Finalize CCI
  5) Wait for the daemon to exit, if this process started it

  Each call must call once.
*/
int io_finalize(void);

#endif /* IO_H */
