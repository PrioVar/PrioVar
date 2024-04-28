import plusFill from '@iconify/icons-eva/plus-fill'
import { Icon } from '@iconify/react'
import {
//  Avatar,
  Button,
//  Card,
//  Checkbox,
  Container,
/*
  Stack,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TablePagination,
  TableRow,
  Typography,
  Box,
  */
} from '@material-ui/core'
// material
//import { useTheme } from '@material-ui/core/styles'
//import { sentenceCase } from 'change-case'
//import { filter } from 'lodash'
//import { useEffect, useState } from 'react'
import { Link as RouterLink } from 'react-router-dom'
import MUIDataTable from 'mui-datatables'
import { UserMoreMenu } from '../../components/_dashboard/user/list'
import HeaderBreadcrumbs from '../../components/HeaderBreadcrumbs'
//import Label from '../../components/Label'
// components
import Page from '../../components/Page'
//import Scrollbar from '../../components/Scrollbar'
//import SearchNotFound from '../../components/SearchNotFound'
// hooks
import useSettings from '../../hooks/useSettings'
//import { deleteUser, getUserList } from '../../redux/slices/user'
// redux
//import { useDispatch, useSelector } from '../../redux/store'
// routes
import { PATH_DASHBOARD } from '../../routes/paths'
import { useUserList } from '../../api/user/list'
import { useManageUser } from '../../api/user/manage'

// ----------------------------------------------------------------------

const MoreMenu = function ({ userId }) {
  const { deleteMutation } = useManageUser({ userId })

  const handleDelete = () => {
    deleteMutation.mutate()
  }

  return <UserMoreMenu onDelete={handleDelete} userName={userId} />
}

export default function UserList() {
  const { themeStretch } = useSettings()
  const { data = [] } = useUserList()

  const COLUMNS = [
    {
      name: 'email',
      label: 'Email',
      options: {
        filter: true,
        sort: true,
      },
    },
    {
      name: 'first_name',
      label: 'First Name',
      options: {
        filter: true,
        sort: true,
      },
    },
    {
      name: 'last_name',
      label: 'Last Name',
      options: {
        filter: true,
        sort: true,
      },
    },
    { name: '', label: '', options: { filter: false, sort: false } },
    { name: '', label: '', options: { filter: false, sort: false } },
    {
      name: 'last_name',
      label: '    ',
      options: {
        filter: true,
        sort: true,
        customBodyRenderLite: (dataIndex) => {
          const row = data[dataIndex]
          if (!row) {
            return null
          }
          return <MoreMenu userId={row.id} />
          // return <SampleSelector fileId={row.id} disabled={row.job_state !== 'DONE'} />
        },
      },
    },
  ]

  return (
    <Page title="User: List | PrioVar">
      <Container maxWidth={themeStretch ? false : 'lg'}>
        <HeaderBreadcrumbs
          heading=""
          // links={[
          //   { name: 'Dashboard', href: PATH_DASHBOARD.root },
          //   { name: 'User', href: PATH_DASHBOARD.user.root },
          //   { name: 'List' },
          // ]}
          action={
            <Button
              variant="contained"
              component={RouterLink}
              to={PATH_DASHBOARD.user.newUser}
              startIcon={<Icon icon={plusFill} />}
            >
              New User
            </Button>
          }
        />
        <MUIDataTable
          title="User List"
          data={data}
          columns={COLUMNS}
          options={{
            print: false,
            download: false,
            viewColumns: false,
            filter: false,
            selectableRows: 'none',
            // sortOrder: { name: 'created_at', direction: 'desc' },
          }}
        />

        {/* <Card>
          <UserListToolbar numSelected={selected.length} filterName={filterName} onFilterName={handleFilterByName} />

          <Scrollbar>
            <TableContainer sx={{ minWidth: 800 }}>
              <Table>
                <UserListHead
                  order={order}
                  orderBy={orderBy}
                  headLabel={TABLE_HEAD}
                  rowCount={userList.length}
                  numSelected={selected.length}
                  onRequestSort={handleRequestSort}
                  onSelectAllClick={handleSelectAllClick}
                />
                <TableBody>
                  {filteredUsers.slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage).map((row) => {
                    const { id, name, role, status, company, avatarUrl, isVerified } = row
                    const isItemSelected = selected.indexOf(name) !== -1

                    return (
                      <TableRow
                        hover
                        key={id}
                        tabIndex={-1}
                        role="checkbox"
                        selected={isItemSelected}
                        aria-checked={isItemSelected}
                      >
                        <TableCell padding="checkbox">
                          <Checkbox checked={isItemSelected} onChange={(event) => handleClick(event, name)} />
                        </TableCell>
                        <TableCell component="th" scope="row" padding="none">
                          <Stack direction="row" alignItems="center" spacing={2}>
                            <Avatar alt={name} src={avatarUrl} />
                            <Typography variant="subtitle2" noWrap>
                              {name}
                            </Typography>
                          </Stack>
                        </TableCell>
                        <TableCell align="left">{company}</TableCell>
                        <TableCell align="left">{role}</TableCell>
                        <TableCell align="left">{isVerified ? 'Yes' : 'No'}</TableCell>
                        <TableCell align="left">
                          <Label
                            variant={theme.palette.mode === 'light' ? 'ghost' : 'filled'}
                            color={(status === 'banned' && 'error') || 'success'}
                          >
                            {sentenceCase(status)}
                          </Label>
                        </TableCell>

                        <TableCell align="right">
                          <UserMoreMenu onDelete={() => handleDeleteUser(id)} userName={name} />
                        </TableCell>
                      </TableRow>
                    )
                  })}
                  {emptyRows > 0 && (
                    <TableRow style={{ height: 53 * emptyRows }}>
                      <TableCell colSpan={6} />
                    </TableRow>
                  )}
                </TableBody>
                {isUserNotFound && (
                  <TableBody>
                    <TableRow>
                      <TableCell align="center" colSpan={6} sx={{ py: 3 }}>
                        <SearchNotFound searchQuery={filterName} />
                      </TableCell>
                    </TableRow>
                  </TableBody>
                )}
              </Table>
            </TableContainer>
          </Scrollbar>

          <TablePagination
            rowsPerPageOptions={[5, 10, 25]}
            component="div"
            count={userList.length}
            rowsPerPage={rowsPerPage}
            page={page}
            onPageChange={handleChangePage}
            onRowsPerPageChange={handleChangeRowsPerPage}
          />
        </Card> */}
      </Container>
    </Page>
  )
}
